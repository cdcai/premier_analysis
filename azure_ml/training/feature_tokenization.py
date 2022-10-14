'''This script merges the feature columns and converts them to ints.'''

# %%
import pandas as pd
import numpy as np
import pickle as pkl
import os
from sklearn.feature_extraction.text import CountVectorizer
from tools import preprocessing as tp
import time

import argparse

########## AZUREML package ######################
from azureml.core import Workspace, Datastore, Dataset
from azureml.core import Run

# %% Setting top-level parameters
MIN_DF = 5
NO_VITALS = False
ADD_DEMOG = True
TIME_UNIT = "dfi"
REVERSE_VOCAB = True
MISA_ONLY = True

# Whether to write the full trimmed sequence file to disk as pqruet
WRITE_PARQUET = False

# Setting the directories
pwd = os.path.abspath(os.path.dirname(__file__))
output_dir = os.path.join(pwd, "..", "output", "")
data_dir = os.path.join(pwd, "..", "data", "data", "")
targets_dir = os.path.join(pwd, "..", "data", "targets", "")
pkl_dir = os.path.join(output_dir, "pkl", "")
parq_dir = os.path.join(output_dir, "parquet", "")

[os.makedirs(path, exist_ok=True) for path in [output_dir,data_dir,parq_dir, pkl_dir,targets_dir]]

ftr_cols = ['vitals', 'bill', 'genlab', 'lab_res', 'proc', 'diag']
demog_vars = ["gender", "hispanic_ind", "age", "race"]
final_cols = ['covid_visit', 'ftrs']


datastore_name = 'edav_dev_ds'
cdh_path = 'exploratory/databricks_ml/mitre_premier/data/'
cdh_path_targets = 'exploratory/databricks_ml/mitre_premier/targets/'
datasets_path_name =  ['vw_covid_id','vw_covid_pat_all','providers']


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

def main() :

    parser = argparse.ArgumentParser("prepare")

    parser.add_argument("--flat_features",type=str)

    parser.add_argument("--trimmed_seq_file",type=str)
    parser.add_argument("--pat_data_file",type=str)
    parser.add_argument("--demog_dict_file",type=str)
    parser.add_argument("--all_ftrs_dict_file",type=str)
    parser.add_argument("--int_seqs_file",type=str)

    ftr_cols = ['vitals', 'bill', 'genlab', 'lab_res', 'proc', 'diag']
    demog_vars = ["gender", "hispanic_ind", "age", "race"]
    final_cols = ['covid_visit', 'ftrs']

    args = parser.parse_args()

    print(args.flat_features)


    ########## Run AZUREML ######################
    run = Run.get_context()
    print("run name:",run.display_name)
    print("run details:",run.get_details())

    ws = run.experiment.workspace
    # retrieve an existing datastore in the workspace by name
    datastore = Datastore.get(ws, datastore_name)

    print("Downloading premier parquet files..")
    for ds_name in datasets_path_name:
        download_files(datastore,ds_name,cdh_path,data_dir)

    print("Download icu targets")
    # Azure data lake storage path
    icu_datapath = os.path.join(cdh_path_targets,'icu_targets.csv')
    datastore_icu_paths = [(datastore, icu_datapath)]
    #load parquet files from # Azure data lake storage
    ds = Dataset.File.from_files(path= datastore_icu_paths)
    ds.download(target_path=targets_dir,overwrite=True,ignore_not_found=True,)

    print("icu targets:", targets_dir)
    print("icu targets:", targets_dir + 'icu_targets.csv')

    ####### SAVE in the Pipeline Data ##############
    os.makedirs(args.trimmed_seq_file, exist_ok=True)
    os.makedirs(args.pat_data_file, exist_ok=True)
    os.makedirs(args.demog_dict_file, exist_ok=True)
    os.makedirs(args.all_ftrs_dict_file, exist_ok=True)
    os.makedirs(args.int_seqs_file, exist_ok=True)


    # %% Read in the pat and ID tables
    pat_df = pd.read_parquet(data_dir + "vw_covid_pat_all/")
    id_df = pd.read_parquet(data_dir + "vw_covid_id/")
    provider = pd.read_parquet(data_dir + "providers/")
    misa_data = pd.read_csv(targets_dir + 'icu_targets.csv')

    # Read in the flat feature file
    # trimmed_seq = pd.read_parquet(output_dir + "parquet/flat_features.parquet")

    #### READING from the Pipeline Data parameters
    trimmed_seq = pd.read_parquet(os.path.join(args.flat_features,"flat_features.parquet"))

    # %% Filter Denom to those identified in MISA case def
    if MISA_ONLY:
        trimmed_seq = trimmed_seq[trimmed_seq.medrec_key.isin(
            misa_data.medrec_key)]

    # Determine unique patients
    n_patients = trimmed_seq["medrec_key"].nunique()

    # Ensure we're sorted
    trimmed_seq.sort_values(["medrec_key", "dfi"], inplace=True)

    # %% Optionally drops vitals and genlab from the features
    if NO_VITALS:
        ftr_cols = ['bill', 'lab_res', 'proc', 'diag']

    # Combining the separate feature columns into one
    trimmed_seq["ftrs"] = (trimmed_seq[ftr_cols].astype(str).replace(
        ["None", "nan"], "").agg(" ".join, axis=1))

    # %% Fitting the vectorizer to the features
    print("Create vocab using CountVectorizer")
    ftrs = [doc for doc in trimmed_seq.ftrs]
    vec = CountVectorizer(ngram_range=(1, 1), min_df=MIN_DF, binary=True)
    vec.fit(ftrs)
    vocab = vec.vocabulary_

    # Saving the index 0 for padding
    for k in vocab.keys():
        vocab[k] += 1

    # Converting the bags of feature strings to integers
    int_ftrs = [[vocab[k] for k in doc.split() if k in vocab.keys()]
                for doc in ftrs]
    trimmed_seq["int_ftrs"] = int_ftrs

    # list of integer sequence arrays split by medrec_key
    int_seqs = [
        df.values for _, df in trimmed_seq.groupby("medrec_key")["int_ftrs"]
    ]

    # Converting to a nested list to keep things clean
    seq_gen = [[seq for seq in medrec] for medrec in int_seqs]

    # %% Optionally add demographics
    print("Optionally add demographics")
    if ADD_DEMOG:
        # Append demog
        trimmed_plus_demog = trimmed_seq.merge(pat_df[["medrec_key"] + demog_vars],
                                            how="left").set_index("medrec_key")

        if "age" in demog_vars:
            trimmed_plus_demog = tp.max_age_bins(trimmed_plus_demog,
                                                bins=np.arange(0, 111, 10))

        # %% Take distinct by medrec
        demog_map = map(lambda name: name + ":" + trimmed_plus_demog[name],
                        demog_vars)
        demog_labeled = pd.concat(demog_map, axis=1)
        raw_demog = demog_labeled.reset_index().drop_duplicates()
        just_demog = raw_demog.groupby("medrec_key").agg(
            lambda x: " ".join(list(set(x))).lower())

        # BUG: Note there are some medrecs with both hispanic=y and hispanic=N
        just_demog["all_demog"] = just_demog[demog_vars].agg(" ".join, axis=1)
        demog_list = [demog for demog in just_demog.all_demog]
        assert just_demog.shape[0] == n_patients, "No funny business"
        demog_vec = CountVectorizer(binary=True, token_pattern=r"(?u)\b[\w:-]+\b")
        demog_vec.fit(demog_list)
        demog_vocab = demog_vec.vocabulary_
        # This allows us to use 0 for padding if we coerce to dense
        for k in demog_vocab.keys():
            demog_vocab[k] += 1
        demog_ints = [[
            demog_vocab[k] for k in doc.split() if k in demog_vocab.keys()
        ] for doc in demog_list]

        # Zip with seq_gen to produce a list of tuples
        seq_gen = [seq for seq in zip(seq_gen, demog_ints)]

        # And saving vocab
        #with open(pkl_dir + "demog_dict.pkl", "wb") as f:
        #    pkl.dump(demog_vocab, f)

        with open(os.path.join(args.demog_dict_file, "demog_dict.pkl"), "wb") as f:
            pkl.dump(demog_vocab, f)

    # === Figuring out which visits were covid visits,
    # and which patients have no covid visits (post-trim)

    cv_dict = dict(zip(pat_df.pat_key, pat_df.covid_visit))
    cv_pats = [[cv_dict[pat_key] for pat_key in np.unique(seq.values)]
            for _, seq in trimmed_seq.groupby("medrec_key").pat_key]

    no_covid = np.where([np.sum(doc) == 0 for doc in cv_pats])[0]

    # With the new trimming, this should never be populated
    assert len(no_covid) == 0

    # Additional sanity check
    assert len(cv_pats) == len(seq_gen) == trimmed_seq.medrec_key.nunique()

    # Writing the trimmed sequences to disk
    if WRITE_PARQUET:
        # trimmed_seq.to_parquet(output_dir + 'parquet/trimmed_seq.parquet') #trimmed_seq_file
        trimmed_seq.to_parquet(os.path.join(args.trimmed_seq_file,'trimmed_seq.parquet'))

    # Save list-of-list-of-lists as pickle
    #with open(pkl_dir + "int_seqs.pkl", "wb") as f:
    #    pkl.dump(seq_gen, f)

    with open(os.path.join(args.int_seqs_file,"int_seqs.pkl"), "wb") as f:
        pkl.dump(seq_gen, f)

    # Freeing up memory
    seq_gen = []

    # Figuring out how many feature bags in each sequence belong
    # to each visit
    pat_lengths = trimmed_seq.groupby(["medrec_key", "pat_key"],
                                    sort=False).pat_key.count()
    pat_lengths = [[n for n in df.values]
                for _, df in pat_lengths.groupby("medrec_key")]

    # %% Making a groupby frame to use below
    grouped_pat_keys = trimmed_seq.groupby("medrec_key").pat_key

    # %% Figuring out whether a patient died after a visit
    died = np.array(["EXPIRED" in status for status in pat_df.disc_status_desc],
                    dtype=np.uint8)
    death_dict = dict(zip(pat_df.pat_key, died))
    pat_deaths = [[death_dict[id] for id in np.unique(df.values)]
                for _, df in grouped_pat_keys]

    # Adding the inpatient variable to the pat dict
    inpat = np.array(pat_df.pat_type == 8, dtype=np.uint8)
    inpat_dict = dict(zip(pat_df.pat_key, inpat))
    pat_inpat = [[inpat_dict[id] for id in np.unique(df.values)]
                for _, df in grouped_pat_keys]

    # %% Adding the ICU indicator
    icu_pats = misa_data[misa_data.icu_visit == "Y"].pat_key
    icu_dict = dict(zip(pat_df.pat_key, [0] * len(pat_df.pat_key)))
    for pat in icu_pats:
        icu_dict.update({pat: 1})
    icu = [[icu_dict[id] for id in np.unique(df.values)]
        for _, df in grouped_pat_keys]

    # %% Adding age at each visit
    age = pat_df.age.values.astype(np.uint8)
    age_dict = dict(zip(pat_df.pat_key, age))
    pat_age = [[age_dict[id] for id in np.unique(df.values)]
            for _, df in grouped_pat_keys]

    # Mixing in the MIS-A targets and Making a lookup for the first case definition
    misa_pt_pats = misa_data[misa_data.misa_filled == 1].pat_key
    misa_pt_dict = dict(zip(pat_df.pat_key, [0] * len(pat_df.pat_key)))
    for pat in misa_pt_pats:
        misa_pt_dict.update({pat: 1})

    misa_pt = [[misa_pt_dict[id] for id in np.unique(df.values)]
            for _, df in grouped_pat_keys]

    #  And finally saving a the pat_keys themselves to facilitate
    # record linkage during analysis
    pat_key = [[num for num in df.values] for _, df in grouped_pat_keys]

    # Rolling things up into a dict for easier saving
    pat_dict = {
        'key': pat_key,
        'age': pat_age,
        'covid': cv_pats,
        'length': pat_lengths,
        'inpat': pat_inpat,
        'outcome': {
            'icu': icu,
            'death': pat_deaths,
            'misa_pt': misa_pt
        }
    }

    # %%
    #with open(pkl_dir + "pat_data.pkl", "wb") as f:
    #    pkl.dump(pat_dict, f)

    with open(os.path.join(args.pat_data_file,"pat_data.pkl"), "wb") as f:
        pkl.dump(pat_dict, f)

    # Optionally reversing the vocab
    if REVERSE_VOCAB:
        vocab = {v: k for k, v in vocab.items()}

    # Saving the updated vocab to disk
    #with open(pkl_dir + "all_ftrs_dict.pkl", "wb") as f:
    #    pkl.dump(vocab, f)

    with open(os.path.join(args.all_ftrs_dict_file,"all_ftrs_dict.pkl"), "wb") as f:
        pkl.dump(vocab, f)


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()

    print("Time total: {}".format(t2 - t1))