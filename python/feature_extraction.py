#%%
import numpy as np
import pandas as pd
import pickle
import json
import ipython_memory_usage.ipython_memory_usage as imu

from importlib import reload

import tools.preprocessing as tp
import tools.multi as tm

imu.start_watching_memory()
# %%Unit of time to use for aggregation
TIME_UNIT = "dfi"

# Setting the file directories
prem_dir = "../data/data/"
out_dir = "../output/"
parq_dir = out_dir + "parquet/"
pkl_dir = out_dir + "pkl/"

# %% Lazily Importing the parquet files
print("")
print("Loading the parquet files...")

pq = tm.parquets_dask(prem_dir, agg_lvl=TIME_UNIT)

# %% Pull all data as dask Df to a dictionary
# and save the feature dictionaries to a pkl
all_data, _ = pq.all_df_to_feat(pkl_dir + "feature_lookup.pkl")

# %% Calculating time from index for each observation (based on agg level)
vitals = pq.get_timing(
    all_data["vitals"],
    day_col="observation_day_number",
    time_col="observation_time_of_day",
)

bill = pq.get_timing(all_data["bill"], day_col="serv_day")

genlab = pq.get_timing(
    all_data["gen_lab"],
    day_col="collection_day_number",
    time_col="collection_time_of_day",
)

lab_res = pq.get_timing(
    all_data["lab_res"], day_col="spec_day_number", time_col="spec_time_of_day"
)

proc = pq.get_timing(all_data["proc"], day_col="proc_day")

diag = pq.get_timing(all_data["diag"])

# %% Aggregating features by day
print("Aggregating the features by day...")
vitals_agg = pq.agg_features(vitals)
bill_agg = pq.agg_features(bill)
genlab_agg = pq.agg_features(genlab)
lab_res_agg = pq.agg_features(lab_res)
proc_agg = pq.agg_features(proc)
diag_agg = pq.agg_features(diag)


# %% Merging all the tables into a single flat file
print("And merging the aggregated tables into a flat file.")
agg = [vitals_agg, bill_agg, genlab_agg, lab_res_agg, proc_agg, diag_agg]
agg_names = ["vitals", "bill", "genlab", "lab_res", "proc", "diag"]
agg_merged = tp.merge_all(agg, on=["pat_key", TIME_UNIT])
agg_merged.columns = ["pat_key", TIME_UNIT] + agg_names

agg_merged = agg_merged.set_index("pat_key")
# Adjusting diag times to be at the end of the visit
# BUG: Figure out to to do this with Dask
# Probably requires LOS column in the id table and adding that
# instead during pq.get_timing

# max_times = agg_merged.groupby("pat_key")[TIME_UNIT].max()
# max_ids = max_times.index.values
# if TIME_UNIT != "dfi":
#     max_dict = dict(zip(max_ids, max_times.values + 1))
# else:
#     max_dict = dict(zip(max_ids, max_times.values))

# base_dict = dict(zip(diag_agg.pat_key, diag_agg[TIME_UNIT]))
# base_dict.update(max_dict)
# diag_agg[TIME_UNIT] = [base_dict[id] for id in diag_agg.pat_key]

# Merging diagnoses with the rest of the columns
# agg_all = tp.merge_all([agg_merged, diag_agg], on=["pat_key", TIME_UNIT])
# agg_all.rename({"ftrs": "dx"}, axis=1, inplace=True)

# %% Adding COVID visit indicator
agg_all = agg_all.merge(
    pq.id[["pat_key", "covid_visit", "medrec_key"]].set_index("pat_key"),
    left_index=True,
    right_index=True,
)

# Reordering the columns
agg_all = agg_all[
    [
        "medrec_key",
        "pat_key",
        TIME_UNIT,
        "vitals",
        "bill",
        "genlab",
        "lab_res",
        "proc",
        "dx",
        "covid_visit",
    ]
]

# %% Sorting by medrec, pat, and time
# HACK: sort_values isn't supported by default, so we will try something else
agg_all = agg_all.map_partitions(
    lambda df: df.sort_values(["medrec_key", "pat_key", TIME_UNIT])
)

# %% Writing a sample of the flat file to disk
samp_ids = agg_all.pat_key.sample(1000)
agg_samp = agg_all[agg_all.pat_key.isin(samp_ids)]
agg_samp.to_csv(out_dir + "samples/agg_samp.csv", index=False)

# %% Writing the flat feature file to disk
agg_all.to_parquet(
    parq_dir + "flat_features/", index=False, row_group_size=5000, engine="pyarrow"
)
