# %%
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

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

# %% Pad sequences for timestep
trimmed_seq.reset_index(drop=False, inplace=True)
trimmed_seq.set_index(["medrec_key"], inplace=True)

# list of np arrays split by medrec_key
df_gen = [df.values for _, df in trimmed_seq["ftrs"].groupby("medrec_key")]

padded_seq = pad_sequences(df_gen, maxlen=time_seq, dtype=object, value="")

# %% Construct a RaggedTensor with the split text data
# (n_medrec, time_seq) -> (n_medrec, time_seq, 1)
X = tf.strings.split(padded_seq[:, :, np.newaxis])
X.shape

# %% Create labels
# TODO
# %%
dataset = tf.data.Dataset.from_tensor_slices(X)