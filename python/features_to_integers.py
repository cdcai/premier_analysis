"""This script merges the feature columns and converts them to ints."""

# %%
import pandas as pd
import numpy as np
import pickle as pkl
import os
import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Setting top-level parameters
MIN_DF = 5
NO_VITALS = True
PAD_BAGS = False
PAD_SEQS = False
PAD_VAL = 0
MAX_TIME = 225
TIME_UNIT = "dfi"

# Whether to write the full trimmed sequence file to disk as parquet
WRITE_PARQUET = False

# Setting the directories
output_dir = os.path.abspath('output/') + '/'
data_dir = os.path.abspath('data/data/') + '/'
targets_dir = os.path.abspath("data/targets") + "/"
pkl_dir = output_dir + 'pkl/'
ftr_cols = ['vitals', 'bill', 'genlab', 'lab_res', 'proc', 'diag']
final_cols = ['covid_visit', 'ftrs']

# %% Read in the pat and ID tables
pat_df = pd.read_parquet(data_dir + "vw_covid_pat_all/")
id_df = pd.read_parquet(data_dir + "vw_covid_id/")
misa_data = pd.read_csv(targets_dir + 'targets.csv', ";")

# Read in the flat feature file
all_features = pd.read_parquet(output_dir + "parquet/flat_features/")

# Determine unique medrec_keys
n_medrec = all_features["medrec_key"].nunique()

# Ensure we're sorted
all_features.sort_values(["medrec_key", "pat_key", "dfi"], inplace=True)

# %% Trim the sequences
# NOTE: This needs to be a little more complex
# to work with the MIS-A targets.
# - IF MIS-A:
#   MIS-A visits are always the first COVID visit
#   which matches, so we take the max follow up
#   from that visit and trim backwards from there
# - If non-MIS-A
#   We take the max follow up time from the most recent
#   COVID visit and trim back from there.

# Filter to only the pat_key where MIS-A took place
# and compute the max F/O time for that visit, indexing on
# medrec key
misa_latest = all_features.merge(
    misa_data,
    how="inner",
    left_on=["medrec_key", "pat_key"],
    right_on=["medrec_key", "first_misa_patkey"
              ]).groupby("medrec_key")[TIME_UNIT].agg(max).rename("max_time")

# Find which medrec_keys are Non-MIS-A where we will take the latest visit of COVID
non_misa_ids = set(all_features.medrec_key).difference(misa_latest.index)

# Pull the latest COVID visit from Non-MIS-A medrec keys
# and find max F/O time
non_misa_latest = all_features.loc[
    all_features.medrec_key.isin(non_misa_ids) & all_features["covid_visit"] ==
    1].groupby("medrec_key")[TIME_UNIT].agg(max).rename("max_time")

# Combine both into lookup dataframe
latest_covid = pd.concat([misa_latest, non_misa_latest], axis=0)

# Sanity check
assert latest_covid.size == n_medrec

# Join into main dataset
trimmed_seq = all_features.merge(latest_covid, on="medrec_key")

# Trim sequences from the last F/O time we allow
# to the earliest we want to pass to the recurrent model
# then drop the max_time col
trimmed_seq = trimmed_seq.loc[(trimmed_seq[TIME_UNIT] <= trimmed_seq.max_time)
                              & (trimmed_seq[TIME_UNIT] >=
                                 (trimmed_seq.max_time - MAX_TIME))].drop(
                                     "max_time", axis=1)

# Optionally drops vitals and genlab from the features
if NO_VITALS:
    ftr_cols = ["bill", "lab_res", "proc", "diag"]

# %% Combining the separate feature columns into one
trimmed_seq["ftrs"] = (trimmed_seq[ftr_cols].astype(str).replace(
    ["None", "nan"], "").agg(" ".join, axis=1))

# %% Fitting the vectorizer to the features
ftrs = [doc for doc in trimmed_seq.ftrs]
vec = CountVectorizer(ngram_range=(1, 1), min_df=MIN_DF, binary=True)
vec.fit(ftrs)
vocab = vec.vocabulary_

# Saving the index 0 for padding
for k in vocab.keys():
    vocab[k] += 1

# %% Saving the updated vocab to disk
with open(pkl_dir + "all_ftrs_dict.pkl", "wb") as f:
    pkl.dump(vocab, f)

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

# Optionally padding the sequence of visits
if PAD_SEQS:
    seq_gen = [l + [[PAD_VAL]] * (MAX_TIME - len(l)) for l in seq_gen]

# Starting to construct the labels part 1: figuring out which visit
# were covid visits, and which patients have no covid visits (post-trim)
cv_dict = dict(zip(pat_df.pat_key, pat_df.covid_visit))
cv_pats = [[cv_dict[pat_key] for pat_key in np.unique(seq.values)]
           for _, seq in trimmed_seq.groupby("medrec_key").pat_key]

no_covid = np.where([np.sum(doc) == 0 for doc in cv_pats])[0]

# With the new trimming, this should never be populated
assert len(no_covid) == 0

# Additional Sanity check
assert len(cv_pats) == len(seq_gen) == trimmed_seq.medrec_key.nunique()

# %% Writing the trimmed sequences to disk
if WRITE_PARQUET:
    trimmed_seq.to_parquet(output_dir + 'parquet/trimmed_seq.parquet')

# Save list-of-list-of-lists as pickle
with open(pkl_dir + "int_seqs.pkl", "wb") as f:
    pkl.dump(seq_gen, f)

# %% Part 2: figuring out how many feature bags in each sequence belong
# to each visit
pat_lengths = trimmed_seq.groupby(["medrec_key", "pat_key"]).pat_key.count()
pat_lengths = [[n for n in df.values]
               for _, df in pat_lengths.groupby("medrec_key")]

# %% Part 3: Figuring out whether a patient died after a visit
died = np.array(["EXPIRED" in status for status in pat_df.disc_status_desc],
                dtype=np.uint8)
death_dict = dict(zip(pat_df.pat_key, died))
pat_deaths = [[death_dict[id] for id in np.unique(df.values)]
              for _, df in trimmed_seq.groupby("medrec_key").pat_key]

# %% Rolling things up into a dict for easier saving
pat_dict = {
    "cv_pats": cv_pats,
    "pat_lengths": pat_lengths,
    "pat_deaths": pat_deaths
}

# Part 4: Mixing in the MIS-A targets

# Making a lookup for the first case definition
misa_pt_pats = misa_data[misa_data.misa_pt == 1].first_misa_patkey
misa_pt_dict = dict(zip(pat_df.pat_key, [0] * len(pat_df.pat_key)))
[misa_pt_dict.update({pat: 1}) for pat in misa_pt_pats]

misa_pt = [[misa_pt_dict[id] for id in np.unique(df.values)]
           for _, df in trimmed_seq.groupby('medrec_key').pat_key]

# And making a lookup for the second case definition
misa_resp_pats = misa_data[misa_data.misa_resp == 1].first_misa_patkey
misa_resp_dict = dict(zip(pat_df.pat_key, [0] * len(pat_df.pat_key)))

for pat in misa_resp_pats:
    misa_resp_dict.update({pat: 1})
misa_resp = [[misa_resp_dict[id] for id in np.unique(df.values)]
             for _, df in trimmed_seq.groupby('medrec_key').pat_key]

# Rolling things up into a dict for easier saving
pat_dict = {
    'covid': cv_pats,
    'length': pat_lengths,
    'death': pat_deaths,
    'misa_pt': misa_pt,
    'misa_resp': misa_resp
}

with open(pkl_dir + "pat_data.pkl", "wb") as f:
    pkl.dump(pat_dict, f)
