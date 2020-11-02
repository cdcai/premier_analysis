#%%
import tools.preprocessing as tp
import tools.multi as tm
import os
import time
# %%Unit of time to use for aggregation
TIME_UNIT = "dfi"

# Setting the file directories
data_dir = 'C:/Users/yle4/code/github/premier_data/'
prem_dir = data_dir + "/data/"
out_dir = data_dir + "output/"
parq_dir = out_dir + "parquet/"
pkl_dir = out_dir + "pkl/"

_ = [os.makedirs(dirs, exist_ok=True) for dirs in [parq_dir, pkl_dir]]
# %% Lazily Importing the parquet files
print("")
print("Loading the parquet files...")

t1 = time.time()
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

diag = pq.get_timing(all_data["diag"], end_of_visit=True)

# %% Aggregating features by day
print("Aggregating the features by day...")
vitals_agg = pq.agg_features(vitals, out_col="vitals")
bill_agg = pq.agg_features(bill, out_col="bill")
genlab_agg = pq.agg_features(genlab, out_col="genlab")
lab_res_agg = pq.agg_features(lab_res, out_col="lab_res")
proc_agg = pq.agg_features(proc, out_col="proc")
diag_agg = pq.agg_features(diag, out_col="diag")


# %% Merging all the tables into a single flat file
print("And merging the aggregated tables into a flat file.")
agg = [vitals_agg, bill_agg, genlab_agg, lab_res_agg, proc_agg, diag_agg]
agg_merged = tm.dask_merge_all(agg, how="outer")

# %% Adding COVID visit indicator

agg_all = agg_merged.reset_index(drop=False)

agg_all = agg_all.join(
    pq.id.result()[["covid_visit", "medrec_key"]],
    how="left",
    on="pat_key",
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
        "diag",
        "covid_visit",
    ]
]

# %% Sorting by medrec, pat, and time
# BUG: sorting in dask isn't really possible, and multiindex support isn't allowed
# so we will just have to make due, or sort after it's persistent
# agg_all = agg_all.map_partitions(lambda df: df.sort_values([TIME_UNIT, "medrec_key"]))

# %% Writing a sample of the flat file to disk
# BUG: This will honestly take more time than just writing it out and sampling will
# so I've commented it out. Maybe it's fine on the supercomputer

# samp_ids = agg_all.pat_key.sample(frac=0.01).compute().tolist()
# agg_samp = agg_all[agg_all.pat_key.isin(samp_ids)]
# agg_samp.to_csv(out_dir + "samples/agg_samp.csv", index=False)

# %% Writing the flat feature file to disk
pq.client.compute(
    agg_all.to_parquet(parq_dir + "flat_features/", write_index=False),
    sync=True,
)

t2 = time.time()

print("Time total: {}".format(t2-t1))