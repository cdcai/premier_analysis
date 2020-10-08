import numpy as np
import pandas as pd
import pickle
import sys
import os

from importlib import reload
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer

from tools import preprocessing as tp

# Setting the file directories
prem_dir = 'data/data/'
out_dir = 'output/'
parq_dir = out_dir + 'parquet/'
pkl_dir = out_dir + 'pkl/'

# Loading the lookup dicts
ftr_dict = pickle.load(open(pkl_dir + 'feature_lookup.pkl', 'rb'))
admon_dict = pickle.load(open(pkl_dir + 'admon_lookup.pkl', 'rb'))
medrec_dict = pickle.load(open(pkl_dir + 'medrec_lookup.pkl', 'rb'))

# Loading the table of previous diagnoses
prev_diag = pd.read_parquet(parq_dir + 'prev_diag.parquet')

# Loading the master patient file
pat = pd.read_parquet(prem_dir + 'vw_covid_pat_indicators.parquet')
icu_dict = dict(zip(pat.pat_key, pat.icu.astype(int)))
death_dict = dict(zip(pat.pat_key, pat.death.astype(int)))
vent_dict = dict(zip(pat.pat_key, pat.vent.astype(int)))
los_dict = dict(zip(pat.pat_key, pat.los.astype(int)))
pats = pat.pat_key.unique()

# Loading the flat feature file
flat_in = pd.read_parquet(parq_dir + 'flat_features.parquet')

# Limiting the features to day-1 visits after February 2020
d1 = flat_in[flat_in.day == 1]
d1 = d1[d1.month >= 2020103]

# Adding the previous diagnoses as extra features
d1['medrec_key'] = [medrec_dict[id] for id in d1.pat_key]
d1 = d1.merge(prev_diag, on='medrec_key', how='left')

# Merging the individual feature columns
ftr_cols = ['vitals', 'oth', 'pharm',
            'lab_bill', 'genlab', 'lab_res',
            'proc', 'ftrs']
ftrs = d1[ftr_cols].astype(str)
ftrs = ftrs.replace(['None', 'nan'], '')
ftrs = ftrs.agg(' '.join, axis=1)

# Vectorizing the merged feature columns
vec = CountVectorizer(ngram_range=(1, 1), binary=True)
X = vec.fit_transform(ftrs)

# Saving the binary features as an npz
save_npz(out_dir + 'npz/features.npz', X)

# Saving the vectorizer vocab for feature lookup later
pickle.dump(vec.vocabulary_, open(pkl_dir + 'vec_vocab.pkl', 'wb'))

# Adding the outcomes to the day-1 data
d1['death'] = [death_dict[id] for id in d1.pat_key]
d1['vent'] = [vent_dict[id] for id in d1.pat_key]
d1['los'] = [los_dict[id] for id in d1.pat_key]

# Saving the day 1 data to parquet
d1.to_parquet(parq_dir + 'd1.parquet', index=False)
