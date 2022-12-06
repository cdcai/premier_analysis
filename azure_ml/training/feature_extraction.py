'''This script organizes each patient's data into a list of lists of features,
ordered by time.
'''

import os
import pickle
import time
from importlib import reload

import pandas as pd

import tools.multi as tm
import tools.preprocessing as tp

import argparse

########## AZUREML package ######################
from azureml.core import Workspace, Datastore, Dataset
from azureml.core import Run

# Unit of time to use for aggregation
TIME_UNIT = 'dfi'

# Whether to limit the output to folks with at least 1 COVID visit
COVID_ONLY = True

# Setting the file directories
#pwd = os.path.abspath(os.path.dirname(__file__))
#prem_dir = os.path.join(pwd, "..", "data", "data", "")
#out_dir = os.path.join(pwd, "..", "output", "")
#parq_dir = os.path.join(out_dir, "parquet", "")
#pkl_dir = os.path.join(out_dir, "pkl", "")
#samp_dir = os.path.join(out_dir, "samples", "")

##### Setting the file directories for AZURE Container
pwd = os.path.abspath(os.path.dirname(__file__))
prem_dir = os.path.join(pwd, "..", "data", "data", "")
out_dir = os.path.join(pwd, "..", "output", "")
parq_dir = os.path.join(out_dir, "parquet", "")
pkl_dir = os.path.join(out_dir, "pkl", "")
samp_dir = os.path.join(out_dir, "samples", "")


datastore_name = 'edav_dev_ds'
cdh_path = 'exploratory/databricks_ml/mitre_premier/data/'
datasets_path_name =  ['vw_covid_pat','vw_covid_id','vw_covid_genlab',
                            'vw_covid_hx_genlab','vw_covid_lab_res','vw_covid_hx_lab_res','vw_covid_vitals',
                            'vw_covid_hx_vitals','vw_covid_bill_lab','vw_covid_bill_pharm','vw_covid_bill_oth',
                            'vw_covid_hx_bill','vw_covid_paticd_diag','vw_covid_paticd_proc','vw_covid_additional_paticd_diag',
                            'vw_covid_additional_paticd_proc','icdcode','vw_covid_pat_all','providers']


[os.makedirs(path, exist_ok=True) for path in [parq_dir, pkl_dir, samp_dir,prem_dir]]



def download_files(datastore,_path_name,src_dir,download_dir):
    print(f"Downloading data :{_path_name}.....")

    # Azure data lake storage path
    patient_datapath = os.path.join(src_dir,_path_name)
    # print(patient_datapath)
    datastore_paths = [(datastore, patient_datapath)]

    #load parquet files from # Azure data lake storage
    ds = Dataset.File.from_files(path= datastore_paths)

    # local path to download data
    download_data_path = os.path.join(download_dir,_path_name)
     # create folder
    os.makedirs(download_data_path,exist_ok=True)

    ds.download(target_path=download_data_path,overwrite=True,ignore_not_found=True)


def main():

    parser = argparse.ArgumentParser("prepare")
    parser.add_argument("--flat_features",type=str)
    parser.add_argument("--feature_lookup",type=str)

    args = parser.parse_args()

    ########## Run AZUREML ######################
    run = Run.get_context()
    print("run name:",run.display_name)
    print("run details:",run.get_details())

    ws = run.experiment.workspace
    # retrieve an existing datastore in the workspace by name
    datastore = Datastore.get(ws, datastore_name)

    #print("Downloading premier parquet files..")
    for ds_name in datasets_path_name:
        download_files(datastore,ds_name,cdh_path,prem_dir)


    # Importing the parquet files
    print('')
    print('Loading the parquet files...')

    pq = tp.load_parquets(prem_dir)

    # Replacing NaN with 0
    pq.id.dropna(axis=0, subset=['days_from_index'], inplace=True)

    # Making some lookup tables to use later
    medrec_dict = dict(
        zip(pq.id.pat_key.astype(int), pq.id.medrec_key.astype(int)))
    day_dict = dict(
        zip(pq.id.pat_key.astype(int), pq.id.days_from_index.astype(int)))
    covid_dict = dict(
        zip(pq.id.pat_key.astype(int), pq.id.covid_visit.astype(int)))

    print('Converting the free-text fields to features...')



    # Vectorizing the single free text fields
    print("genlab features extraction..")
    genlab, genlab_dict = tp.df_to_features(
        pq.genlab,
        feature_prefix='genl',
        text_col='lab_test_loinc_desc',
        time_cols=['collection_day_number', 'collection_time_of_day'],
        replace_col='lab_test_result',
        num_col='numeric_value')
    pq.genlab = []

    print("# vitals:", len(pq.vitals))


    print("vitals features extraction..")
    vitals, v_dict = tp.df_to_features(
        pq.vitals,
        feature_prefix='vtl',
        text_col='lab_test',
        time_cols=['observation_day_number', 'observation_time_of_day'],
        num_col='test_result_numeric_value')
    pq.vitals = []

    print("billing features extraction..")
    bill, bill_dict = tp.df_to_features(pq.bill,
                                        feature_prefix='bill',
                                        text_col='std_chg_desc',
                                        time_cols=['serv_day'])
    pq.bill = []

    print("procesure features extraction..")
    proc, proc_dict = tp.df_to_features(pq.proc,
                                        feature_prefix='proc',
                                        text_col='icd_code',
                                        time_cols=['proc_day'])
    pq.proc = []

    print("diagnosis features extraction..")
    diag, diag_dict = tp.df_to_features(pq.diag,
                                        feature_prefix='dx',
                                        text_col='icd_code')
    pq.diag = []

    # Dropping pat_keys that won't have a days_from_index
    bill = bill.merge(pq.id.pat_key, how='right')

    print("lab feature features extraction..")
    # Vectorizing the microbiology lab results
    lab_text = pq.lab_res.test.astype(str)
    lab_text = lab_text + ' ' + pq.lab_res.observation.astype(str)
    lab_text = pd.DataFrame(lab_text, columns=['text'])
    lab_res = pd.concat([pq.lab_res, lab_text], axis=1)
    lab_res, lab_res_dict = tp.df_to_features(
        lab_res,
        feature_prefix='lbrs',
        text_col='text',
        time_cols=['spec_day_number', 'spec_time_of_day'])

    # Freeing up the last bit of memory memory
    pq = []

    # Combining the feature dicts and saving to disk
    dicts = [
        v_dict, bill_dict, genlab_dict, proc_dict, diag_dict, lab_res_dict
    ]
    ftr_dict = dict(
        zip(tp.flatten([d.keys() for d in dicts]),
            tp.flatten([d.values() for d in dicts])))

    # Calculating days and minutes from index for each observation
    vitals = tm.get_times(vitals, day_dict, 'observation_day_number',
                          'observation_time_of_day')
    genlab = tm.get_times(genlab, day_dict, 'collection_day_number',
                          'collection_time_of_day')
    lab_res = tm.get_times(lab_res, day_dict, 'spec_day_number',
                           'spec_time_of_day')
    bill = tm.get_times(bill, day_dict, 'serv_day')
    proc = tm.get_times(proc, day_dict, 'proc_day')
    diag = tm.get_times(diag, day_dict)

    # Aggregating features by day
    print('Aggregating the features by day...')
    vitals_agg = tp.agg_features(vitals, TIME_UNIT)
    bill_agg = tp.agg_features(bill, TIME_UNIT)
    genlab_agg = tp.agg_features(genlab, TIME_UNIT)
    lab_res_agg = tp.agg_features(lab_res, TIME_UNIT)
    proc_agg = tp.agg_features(proc, TIME_UNIT)
    diag_agg = tp.agg_features(diag, TIME_UNIT)

    # Merging all the tables into a single flat file
    print('And merging the aggregated tables into a flat file.')
    agg = [vitals_agg, bill_agg, genlab_agg, lab_res_agg, proc_agg]
    agg_names = ['vitals', 'bill', 'genlab', 'lab_res', 'proc']
    agg_merged = tp.merge_all(agg, on=['pat_key', TIME_UNIT])
    agg_merged.columns = ['pat_key', TIME_UNIT] + agg_names

    # Adjusting diag times to be at the end of the visit
    max_times = agg_merged.groupby('pat_key')[TIME_UNIT].max()
    max_ids = max_times.index.values
    if TIME_UNIT != 'dfi':
        max_dict = dict(zip(max_ids, max_times.values + 1))
    else:
        max_dict = dict(zip(max_ids, max_times.values))

    base_dict = dict(zip(diag_agg.pat_key, diag_agg[TIME_UNIT]))
    base_dict.update(max_dict)
    diag_agg[TIME_UNIT] = [base_dict[id] for id in diag_agg.pat_key]

    # Merging diagnoses with the rest of the columns
    agg_all = tp.merge_all([agg_merged, diag_agg], on=['pat_key', TIME_UNIT])
    agg_all.rename({'ftrs': 'diag'}, axis=1, inplace=True)

    # Adding COVID visit indicator
    agg_all['covid_visit'] = [covid_dict[id] for id in agg_all.pat_key]

    # And adding medrec key
    agg_all['medrec_key'] = [medrec_dict[id] for id in agg_all.pat_key]

    # Reordering the columns
    agg_all = agg_all[[
        'medrec_key',
        'pat_key',
        TIME_UNIT,
        'vitals',
        'bill',
        'genlab',
        'lab_res',
        'proc',
        'diag',
        'covid_visit',
    ]]

    # Sorting by medrec, pat, and time
    agg_all.sort_values(['medrec_key', 'pat_key', TIME_UNIT], inplace=True)

    # Optionally getting rid of non-COVID patients; i'm sure there's a more
    # efficient way of doing this, but I can't figure it out.
    if COVID_ONLY:
        total_covid = agg_all.groupby('medrec_key')['covid_visit'].sum()
        total_dict = dict(zip(agg_all.medrec_key.unique(), total_covid > 0))
        covid_medrec = [total_dict[id] for id in agg_all.medrec_key]
        agg_all = agg_all.iloc[covid_medrec, :]

    # Writing a sample of the flat file to disk
    samp_ids = agg_all.pat_key.sample(1000)
    agg_samp = agg_all[agg_all.pat_key.isin(samp_ids)]
    agg_samp.to_csv(samp_dir + 'agg_samp.csv', index=False)

    # Writing the flat feature file to disk
    agg_all.to_parquet(parq_dir + 'flat_features.parquet', index=False)

    # And saving the feature dict to disk
    pickle.dump(ftr_dict, open(pkl_dir + 'feature_lookup.pkl', 'wb'))

    ####### SAVE in the Pipeline Data ##############
    os.makedirs(args.flat_features, exist_ok=True)
    os.makedirs(args.feature_lookup, exist_ok=True)

    agg_all.to_parquet(os.path.join(args.flat_features,'flat_features.parquet') , index=False)
    pickle.dump(ftr_dict, open(os.path.join(args.feature_lookup, 'feature_lookup.pkl'), 'wb'))


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()

    print("Time total: {}".format(t2 - t1))
