# Databricks notebook source
'''This script takes the full list of lists of visits and prepares them for
modeling, e.g., by cutting them to specific lengths and specifying their
labels.
'''

import numpy as np
import pandas as pd
import pickle as pkl
import argparse
import os

from importlib import reload
from multiprocessing import Pool
from tools import preprocessing as tp

# COMMAND ----------

# Set up Azure storage connection
spark.conf.set("fs.azure.account.auth.type.davsynapseanalyticsdev.dfs.core.windows.net", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type.davsynapseanalyticsdev.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id.davsynapseanalyticsdev.dfs.core.windows.net", dbutils.secrets.get(scope="dbs-scope-CDH", key="apps-client-id"))
spark.conf.set("fs.azure.account.oauth2.client.secret.davsynapseanalyticsdev.dfs.core.windows.net", dbutils.secrets.get(scope="dbs-scope-CDH", key="apps-client-secret"))
spark.conf.set("fs.azure.account.oauth2.client.endpoint.davsynapseanalyticsdev.dfs.core.windows.net", dbutils.secrets.get(scope="dbs-scope-CDH", key="apps-tenant-id-endpoint"))

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# COMMAND ----------

def find_cutpoints(visit_type, visit_length, tail=1, origin=0, how='first'):
    '''Figures out where to cut each patient's sequence of visits.
    
    See tools.preprocessing for full docstring.
    '''
    covid_idx = np.where(np.array(visit_type) == 1)[0]
    first = np.min(covid_idx)
    first_end = np.sum(visit_length[0:first], dtype=np.uint16) + tail
    last = np.max(covid_idx)
    last_end = np.sum(visit_length[0:last], dtype=np.uint16) + tail

    if how == 'first':
        if origin != 0:
            origin = np.maximum(0, first_end - origin)
        return (origin, first_end), first
    elif how == 'last':
        if origin != 0:
            origin = np.maximum(0, last_end - origin)
        return (origin, last_end), last
    elif how == 'both':
        return (first_end, last_end), last


def trim_sequence(inputs, labels, cuts):
    '''Trims the sequences of visits according to find_cutpoints.
    
    See tools.preprocessing for full docstring.
    '''
    in_start, in_end = cuts[0][0], cuts[0][1]
    label_id = cuts[1]
    targets = [l[label_id] for l in labels]
    return inputs[0][in_start:in_end], inputs[1], targets


def flatten(l):
    if type(l) != type([]):
        return l
    if type(l[0]) == type([]):
        return [item for sublist in l for item in sublist]
    else:
        return l

# COMMAND ----------

# MAGIC %md
# MAGIC ```
# MAGIC   parser = argparse.ArgumentParser()
# MAGIC   parser.add_argument('--cut_method',
# MAGIC                       type=str,
# MAGIC                       default='first',
# MAGIC                       choices=['first', 'last', 'both'],
# MAGIC                       help='which COVID visit(s) to use as bookends')
# MAGIC   parser.add_argument('--horizon',
# MAGIC                       type=int,
# MAGIC                       default=1,
# MAGIC                       help='prediction horizon for the target visit')
# MAGIC   parser.add_argument('--max_seq',
# MAGIC                       type=int,
# MAGIC                       default=225,
# MAGIC                       help='max number of days to include')
# MAGIC   parser.add_argument(
# MAGIC       '--exclude_icu',
# MAGIC       type=bool,
# MAGIC       default=True,
# MAGIC       help='whether to exclude patients in the ICU before the\
# MAGIC                        prediction horizon')
# MAGIC   parser.add_argument('--out_dir',
# MAGIC                       type=str,
# MAGIC                       default='output/',
# MAGIC                       help='output directory')
# MAGIC   parser.add_argument('--data_dir',
# MAGIC                       type=str,
# MAGIC                       default='..data/data/',
# MAGIC                       help='path to the Premier data')
# MAGIC   parser.add_argument('--min_age',
# MAGIC                       type=int,
# MAGIC                       default=18,
# MAGIC                       help='minimum age of patients to include')
# MAGIC   parser.add_argument('--max_age',
# MAGIC                       type=int,
# MAGIC                       default=120,
# MAGIC                       help='max age of patients to include')
# MAGIC   parser.add_argument('--write_df',
# MAGIC                       type=bool,
# MAGIC                       default=True,
# MAGIC                       help='whether to write patient data to a DF')
# MAGIC   parser.add_argument('--processes',
# MAGIC                       type=int,
# MAGIC                       default=8,
# MAGIC                       help='number of processes to use for multiprocessing')
# MAGIC   args = parser.parse_args()
# MAGIC 
# MAGIC   # Setting the globals
# MAGIC   CUT_METHOD = args.cut_method
# MAGIC   HORIZON = args.horizon
# MAGIC   MAX_SEQ = args.max_seq
# MAGIC   EXCLUDE_ICU = args.exclude_icu
# MAGIC   MIN_AGE = args.min_age
# MAGIC   WRITE_DF = args.write_df
# MAGIC   PROCESSES = args.processes if args.processes != -1 else None
# MAGIC   
# MAGIC   # Setting the directories
# MAGIC   output_dir = os.path.abspath(args.out_dir)
# MAGIC   data_dir = os.path.abspath(args.data_dir)
# MAGIC   pkl_dir = os.path.join(output_dir, "pkl")
# MAGIC ```

# COMMAND ----------

# Setting the globals
CUT_METHOD = 'first'
HORIZON = 1
MAX_SEQ = 225
EXCLUDE_ICU = False
MIN_AGE = 18
WRITE_DF = True
PROCESSES = 8

# COMMAND ----------

# Setting the directories
output_dir = '/dbfs/home/tnk6/premier_output/'
data_dir = '/dbfs/home/tnk6/premier/'
pkl_dir = os.path.join(output_dir, "pkl")

# COMMAND ----------

with open(os.path.join(pkl_dir, 'int_seqs.pkl'), 'rb') as f:
    int_seqs_compare = pkl.load(f)
len(int_seqs_compare)

# COMMAND ----------

int_seqs_compare

# COMMAND ----------

# Reading in the full dataset
#with open(os.path.join(pkl_dir, 'int_seqs.pkl'), 'rb') as f:
#    int_seqs = pkl.load(f)

#with open(os.path.join(pkl_dir, 'pat_data.pkl'), 'rb') as f:
#    pat_data = pkl.load(f)

#with open(os.path.join(pkl_dir, "all_ftrs_dict.pkl"), "rb") as f:
#    vocab = pkl.load(f)

#with open(os.path.join(pkl_dir, "feature_lookup.pkl"), "rb") as f:
#    all_feats = pkl.load(f)

# this works!
#with open(os.path.join(pkl_dir, 'int_seqs_fromdelta.pkl'), 'rb') as f:
#    int_seqs = pkl.load(f)
int_seqs = tp.read_table(data_dir,"interim_int_seqs_pkl")  
int_seqs = int_seqs.values.tolist() 
with open(os.path.join(pkl_dir, 'pat_data_fromdelta.pkl'), 'rb') as f:
    pat_data = pkl.load(f)
vocab = tp.read_table(data_dir,"all_ftrs_dict_pkl")
vocab = dict(vocab.values)
all_feats = tp.read_table(data_dir,"intertim_feature_lookup")
all_feats = dict(all_feats.values)

# Total number of patients
n_patients = len(int_seqs)
print(n_patients)

# COMMAND ----------

int_seqs

# COMMAND ----------

# Trimming the day sequences
with Pool(processes=PROCESSES) as p:
    # Finding the cut points for the sequences
    find_input = [(pat_data['covid'][i], pat_data['length'][i], HORIZON,
                   MAX_SEQ, CUT_METHOD) for i in range(n_patients)]
    cut_points = p.starmap(find_cutpoints, find_input)

    # Trimming the inputs and outputs to the right length
    outcomes = list(pat_data['outcome'].keys())
    outcome_list = [list(pat_data['outcome'][o]) for o in outcomes]
    trim_input = [(int_seqs[i], [l[i]
                                 for l in outcome_list], cut_points[i])
                  for i in range(n_patients)]
    trim_out = p.starmap(trim_sequence, trim_input)

    # Figuring out who has at least 1 more day after the horizon
    keepers = [
        pat_data['length'][i][cut_points[i][1]] > HORIZON
        and pat_data['inpat'][i][cut_points[i][1]] == 1
        and pat_data['age'][i][cut_points[i][1]] >= MIN_AGE
        for i in range(n_patients)
    ]

    # Optionally adding other exclusion criteria
    if EXCLUDE_ICU:
        exclusion_strings = ['ICU', 'TCU', 'STEP DOWN']
        rev_vocab = {v: k for k, v in vocab.items()}
        exclusion_codes = [
            k for k, v in all_feats.items()
            if any(s in v for s in exclusion_strings)
        ]
        exclusion_ftrs = [
            rev_vocab[code] for code in exclusion_codes
            if code in rev_vocab.keys()
        ]
        lookback = [np.min((len(l), HORIZON)) for l in trim_out]
        first_days = [
            trim_out[i][0][-lookback[i]] for i in range(n_patients)
        ]

        # Optionally flattening the pre-horizon code lists;
        # Note: this may or may not work
        if HORIZON > 1:
            first_days = [flatten(l) for l in first_days]

        no_excl = [
            len(np.intersect1d(exclusion_ftrs, l)) == 0 for l in first_days
        ]
        keepers = [keepers[i] and no_excl[i] for i in range(n_patients)]

    # Keeping the keepers and booting the rest
    trim_out = [trim_out[i] for i in range(n_patients) if keepers[i]]

    # Making a DF with the pat-level data to link for analysis later
    if WRITE_DF:
        cohort = [[
            pat_data['key'][i][cut_points[i][1]],
            pat_data['age'][i][cut_points[i][1]],
            pat_data['length'][i][cut_points[i][1]],
            pat_data['outcome']['misa_pt'][i][cut_points[i][1]],
            pat_data['outcome']['icu'][i][cut_points[i][1]],
            pat_data['outcome']['death'][i][cut_points[i][1]]
        ] for i in range(n_patients) if keepers[i]]
        cohort_df = pd.DataFrame(cohort)
        cohort_df.columns = [
            'key', 'age', 'length', 'misa_pt', 'icu', 'death'
        ]
        #cohort_df.to_csv(os.path.join(output_dir, "cohort.csv"), index=False)
        tmp_to_save = spark.createDataFrame(cohort_df)
        tmp_to_save.write.mode("overwrite").format("delta").saveAsTable("tnk6_demo.interim_cohort_csv")

# COMMAND ----------

outcomes = list(pat_data['outcome'].keys())
outcome_list = [list(pat_data['outcome'][o]) for o in outcomes]
outcomes

# COMMAND ----------

# Saving the trimmed sequences to disk
with open(os.path.join(pkl_dir, 'trimmed_seqs_fromdelta.pkl'), 'wb') as f:
    pkl.dump(trim_out, f)

# COMMAND ----------

tmp_to_save = pd.DataFrame(trim_out)
tmp_to_save = spark.createDataFrame(tmp_to_save)
tmp_to_save.write.mode("overwrite").format("delta").saveAsTable("tnk6_demo.interim_trimmed_seqs_pkl")

# COMMAND ----------

display(tmp_to_save)

# COMMAND ----------


