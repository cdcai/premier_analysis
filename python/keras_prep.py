'''This script merges the feature columns and converts them to ints.'''

import pandas as pd
import numpy as np
import pickle as pkl
import os

from sklearn.feature_extraction.text import CountVectorizer


# Setting up top-level parameters
MIN_DF = 5
NO_VITALS = True
PAD_BAGS = False
PAD_SEQS = False
PAD_VAL = 0
MAX_TIME = 225

# Setting up the directories
output_dir = os.path.abspath('../output/') + '/'
data_dir = os.path.abspath('../data/data/') + '/'
pkl_dir = output_dir + 'pkl/'
ftr_cols = ['vitals', 'bill', 'genlab',
            'lab_res', 'proc', 'diag']
final_cols = ['covid_visit', 'ftrs']

# Read in the pat and ID tables
pat_df = pd.read_parquet(data_dir + 'vw_covid_pat_all/')
id_df = pd.read_parquet(data_dir + 'vw_covid_id/') 

# Read in the flat feature file
all_features = pd.read_parquet(output_dir + 'parquet/flat_features.parquet')

# Determine unique medrec_keys
n_medrec = all_features['medrec_key'].nunique()

# Ensure we're sorted
all_features.sort_values(['medrec_key', 'pat_key', "dfi"], inplace=True)

# List of covid visit indicator arrays split by medrec_key
covid_visits = [df.values for _, df
                in all_features.groupby('medrec_key')['covid_visit']]

# Figuring out which visit was the first for COVID;
covid_idx = [np.where(arr == 1)[0] for arr in covid_visits]
first = [idx.min() for idx in covid_idx]
last = [idx.max() for idx in covid_idx]

# Trim the sequences
trimmed_seq = all_features.groupby(['medrec_key']).tail(MAX_TIME)
trimmed_seq.drop_duplicates(inplace=True)

# Optionally drops vitals and genlab from the features
if NO_VITALS:
    ftr_cols = ['bill', 'lab_res', 'proc', 'diag']

# Combining the separate feature columns into one
trimmed_seq['ftrs'] = (
    trimmed_seq[ftr_cols]
    .astype(str)
    .replace(['None', 'nan'], '')
    .agg(' '.join, axis=1)
)

# Fitting the vectorizer to the features
ftrs = [doc for doc in trimmed_seq.ftrs]
vec = CountVectorizer(ngram_range=(1, 1),
                      min_df=MIN_DF,
                      binary=True)
vec.fit(ftrs)
vocab = vec.vocabulary_

# Saving the index 0 for padding
for k in vocab.keys():
    vocab[k] += 1

# Saving the updated vocab to disk
with open(pkl_dir + 'all_ftrs_dict.pkl', 'wb') as f:
    pkl.dump(vocab, f)
    f.close()

# Converting the bags of feature strings to integers
int_ftrs = [[vocab[k] for k in doc.split() if k in vocab.keys()] 
            for doc in ftrs]
trimmed_seq['int_ftrs'] = int_ftrs

# list of integer sequence arrays split by medrec_key
int_seqs = [df.values for _, df 
            in trimmed_seq.groupby('medrec_key')['int_ftrs']]

# Converting to a nested list to keep things clean
seq_gen = [[seq for seq in medrec] for medrec in int_seqs]

# Optionally padding the sequence of visits
if PAD_SEQS:
    seq_gen = [l + [[PAD_VAL]]*(MAX_TIME - len(l)) for l in seq_gen]

# Pickling the sequences of visits for loading in the modeling script
with open(pkl_dir + 'int_seqs.pkl', 'wb') as f:
    pkl.dump(seq_gen, f)
    f.close()

# Constructing some of the targets
died = np.array(['EXPIRED' in status for status in pat_df.disc_status_desc],
                dtype=np.uint8)
death_dict = dict(zip(pat_df.pat_key, died))
pat_died = [
