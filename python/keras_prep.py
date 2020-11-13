'''This script merges the feature columns and converts them to ints.'''

import pandas as pd
import numpy as np
import pickle as pkl

from sklearn.feature_extraction.text import CountVectorizer


# Setting up top-level parameters
MIN_DF = 5
NO_VITALS = True
PAD_BAGS = False
PAD_SEQS = False
PAD_VAL = 0
MAX_TIME = 225

# Setting up the directories
output_dir = "../output/"
data_dir = "../data/data/"
pkl_dir = output_dir + "pkl/"
ftr_cols = ["vitals", "bill", "genlab", "lab_res", "proc", "diag"]
final_cols = ["covid_visit", "ftrs"]

# Read in all data
all_features = pd.read_parquet(output_dir + "parquet/flat_features.parquet")

# Determine unique medrec_keys
n_medrec = all_features["medrec_key"].nunique()

# Ensure we're sorted
all_features.set_index(["medrec_key", "pat_key", "dfi"], inplace=True)
all_features.sort_index(inplace=True)

# Trim the sequences
<<<<<<< HEAD
trimmed_seq = all_features.groupby(['medrec_key']).tail(MAX_TIME)
=======
trimmed_seq = all_features.groupby(["medrec_key"]).tail(MAX_TIME)
>>>>>>> master
trimmed_seq.drop_duplicates(inplace=True)

# Optionally drops vitals and genlab from the features
if NO_VITALS:
    ftr_cols = ["bill", "lab_res", "proc", "diag"]

# Combining the separate feature columns into one
trimmed_seq["ftrs"] = (
    trimmed_seq[ftr_cols].astype(str).replace(["None", "nan"], "").agg(" ".join, axis=1)
)

# Resetting the index
trimmed_seq.reset_index(drop=False, inplace=True)
trimmed_seq.set_index(["medrec_key"], inplace=True)

# Fitting the vectorizer to the features
ftrs = [doc for doc in trimmed_seq.ftrs]
<<<<<<< HEAD
vec = CountVectorizer(ngram_range=(1, 1),
                      min_df=MIN_DF,
                      binary=True)
=======
vec = CountVectorizer(ngram_range=(1, 1), min_df=5, binary=True)
>>>>>>> master
vec.fit(ftrs)
vocab = vec.vocabulary_

# Adding 1 to the vocab indices to 0 can be saved for padding (if needed)
for k in vocab.keys():
    vocab[k] += 1

# Saving the updated vocab to disk
with open(pkl_dir + "all_ftrs_dict.pkl", "wb") as f:
    pkl.dump(vocab, f)
    f.close()

# Converting the bags of feature strings to integers
int_ftrs = [[vocab[k] for k in doc.split() if k in vocab.keys()] for doc in ftrs]
trimmed_seq["int_ftrs"] = int_ftrs

# list of np arrays split by medrec_key
<<<<<<< HEAD
int_seqs = [df.values for _, df 
            in trimmed_seq['int_ftrs'].groupby('medrec_key')]
=======
int_seqs = [df.values for _, df in trimmed_seq["int_ftrs"].groupby("medrec_key")]
>>>>>>> master

# Converting to a nested list to keep things clean
seq_gen = [[seq for seq in medrec] for medrec in int_seqs]

# Optionally padding the sequence of visits
if PAD_SEQS:
    seq_gen = [l + [[PAD_VAL]] * (MAX_TIME - len(l)) for l in seq_gen]

# Pickling the sequences of visits for loading in the modeling script
<<<<<<< HEAD
with open(pkl_dir + 'int_seqs.pkl', 'wb') as f:
=======
with open(pkl_dir + "X.pkl", "wb") as f:
>>>>>>> master
    pkl.dump(seq_gen, f)
    f.close()
