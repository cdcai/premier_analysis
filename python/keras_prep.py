# %%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import pickle

# %%
output_dir = "../output/"
data_dir = "../data/data/"
ftr_cols = ["vitals", "bill", "genlab", "lab_res", "proc", "diag"]
final_cols = ["covid_visit", "ftrs"]

# Max time steps
time_seq = 225

# %% Read in all data
all_features = pd.read_parquet(output_dir + "parquet/flat_features/")
pat_data = pd.read_parquet(data_dir + "vw_covid_pat/")

with open(output_dir + "pkl/feature_lookup.pkl", "rb") as file:
    feature_lookup = pickle.load(file)
# %% DEBUG: Filter to only a few keys to test
# med_key = pat_data.head(2)["medrec_key"].tolist()

# all_features = all_features[all_features.loc[:, "medrec_key"].isin(med_key)]
# %% determine unique medrec_keys
n_medrec = all_features["medrec_key"].nunique()

# %% Ensure we're sorted
all_features.set_index(["medrec_key", "pat_key", "dfi"], inplace=True)
all_features.sort_index(inplace=True)

# %% trim the sequences
# Here we're taking only the most recent times leading up to final covid visit
# as determined by time_seq. We will then have to fill in the ones which don't
# have full observation length or pass as a ragged tensor (but the latter is not as computationally efficient)
trimmed_seq = all_features.groupby(["medrec_key"]).tail(time_seq)
trimmed_seq.drop_duplicates()

trimmed_seq

# %% Aggregate all feature tokens into a single col
# which we will then vectorize and use in an embedding layer
trimmed_seq["ftrs"] = (
    trimmed_seq.loc[:, ftr_cols]
    .astype(str)
    .replace(["None", "nan"], "")
    .agg(" ".join, axis=1)
)

# %% Remove other feature cols we won't need
trimmed_seq.drop(ftr_cols, axis=1, inplace=True)

# %%
trimmed_seq.reset_index(drop=False, inplace=True)
trimmed_seq.set_index(["medrec_key"], inplace=True)

trimmed_seq

# %% Vectorize tokens
# NOTE: We could use n-grams here also, I'll just use BoW
tokenizer = Tokenizer()
tokenizer.fit_on_texts(trimmed_seq["ftrs"])

trimmed_seq["seq"] = tokenizer.texts_to_sequences(trimmed_seq["ftrs"])

# %% produce our sequences
# NOTE: First we pad the token sequences so we can stack
# then we pad the timesteps to the max length

trimmed_seq["padded"] = pad_sequences(
    trimmed_seq.loc[:, "seq"].tolist(), dtype="object"
).tolist()

df_gen = [np.stack(df.values) for _, df in trimmed_seq["seq"].groupby("medrec_key")]

padded_seq = pad_sequences(df_gen, maxlen=time_seq)
# %% Just to make sure we've done it
# Should be (n_medrec, time_seq, token-len)
padded_seq.shape
