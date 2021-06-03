'''This script merges the feature columns and converts them to ints.'''

import pandas as pd
import numpy as np
import pickle as pkl
import os

from sklearn.feature_extraction.text import CountVectorizer


# Setting top-level parameters
MIN_DF = 5
NO_VITALS = False
ADD_DEMOG = True
TIME_UNIT = "dfi"
REVERSE_VOCAB = True
MISA_ONLY = True

# Whether to write the full trimmed sequence file to disk as pqruet
WRITE_PARQUET = False

# Setting the directories
output_dir = os.path.abspath('./output/') + '/'
data_dir = os.path.abspath('../data/data/') + '/'
targets_dir = os.path.abspath('../data/targets/') + '/'
pkl_dir = output_dir + 'pkl/'
ftr_cols = ['vitals', 'bill', 'genlab', 'lab_res', 'proc', 'diag']
final_cols = ['covid_visit', 'ftrs']

# Read in the pat and ID tables
pat_df = pd.read_parquet(data_dir + "vw_covid_pat_all/")
id_df = pd.read_parquet(data_dir + "vw_covid_id/")
misa_data = pd.read_csv(targets_dir + 'icu_targets.csv')

# Read in the flat feature file
trimmed_seq = pd.read_parquet(output_dir + "parquet/flat_features.parquet")

# Filter Denom to those identified in MISA case def
if MISA_ONLY:
    trimmed_seq = trimmed_seq[trimmed_seq.medrec_key.isin(
        misa_data.medrec_key)]

# Determine unique patients
n_patients = trimmed_seq["medrec_key"].nunique()

# Ensure we're sorted
trimmed_seq.sort_values(["medrec_key", "pat_key", "dfi"], inplace=True)

# Optionally drops vitals and genlab from the features
if NO_VITALS:
    ftr_cols = ['bill', 'lab_res', 'proc', 'diag']

# Combining the separate feature columns into one
trimmed_seq["ftrs"] = (trimmed_seq[ftr_cols].astype(str).replace(
    ["None", "nan"], "").agg(" ".join, axis=1))

# Fitting the vectorizer to the features
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

# Optionally add demographics
if ADD_DEMOG:
    demog_vars = ["gender", "hispanic_ind", "race"]
    
    # Append demog
    trimmed_plus_demog = trimmed_seq.merge(pat_df[["medrec_key"] + demog_vars],
                                           how="left").set_index("medrec_key")
    
    # Take distinct by medrec
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
    demog_vec = CountVectorizer(binary=True, token_pattern=r"(?u)\b[\w:]+\b")
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
    with open(pkl_dir + "demog_dict.pkl", "wb") as f:
        pkl.dump(demog_vocab, f)

# Figuring out which visit were covid visits,
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
    trimmed_seq.to_parquet(output_dir + 'parquet/trimmed_seq.parquet')

# Save list-of-list-of-lists as pickle
with open(pkl_dir + "int_seqs.pkl", "wb") as f:
    pkl.dump(seq_gen, f)

# Freeing up memory
seq_gen = []

# Figuring out how many feature bags in each sequence belong
# to each visit
pat_lengths = trimmed_seq.groupby(["medrec_key", "pat_key"]).pat_key.count()
pat_lengths = [[n for n in df.values]
               for _, df in pat_lengths.groupby("medrec_key")]

# Making a groupby frame to use below
grouped_pat_keys = trimmed_seq.groupby("medrec_key").pat_key

# Figuring out whether a patient died after a visit
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

# Adding the ICU indicator
icu_pats = misa_data[misa_data.icu_visit == 1].pat_key
icu_dict = dict(zip(pat_df.pat_key, [0] * len(pat_df.pat_key)))
for pat in icu_pats:
    icu_dict.update({pat: 1})
icu = [[icu_dict[id] for id in np.unique(df.values)]
        for _, df in grouped_pat_keys]

# Adding age at each visit
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

#  Making a lookup for the multiclass labels
misa_multi_df = trimmed_seq[["medrec_key",
                             "pat_key"]].set_index("medrec_key").join(
                                 misa_data[["medrec_key",
                                            "status"]].set_index("medrec_key"))

misa_multi_df = misa_multi_df.drop_duplicates().drop(
    "pat_key", axis=1).reset_index().groupby("medrec_key")

misa_multi = [df.status.to_list() for _, df in misa_multi_df]

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
    'icu': icu,
    'death': pat_deaths,
    'misa_pt': misa_pt,
    'multi_class': misa_multi
}

with open(pkl_dir + "pat_data.pkl", "wb") as f:
    pkl.dump(pat_dict, f)

# Optionally reversing the vocab
if REVERSE_VOCAB:
    vocab = {v: k for k, v in vocab.items()}

# Saving the updated vocab to disk
with open(pkl_dir + "all_ftrs_dict.pkl", "wb") as f:
    pkl.dump(vocab, f)
