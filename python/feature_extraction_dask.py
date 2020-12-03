# %%
import tools.preprocessing as tp
import tools.dask_processing as tm
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import os
import time

# %% Unit of time to use for aggregation
TIME_UNIT = "dfi"
ON_LIZA = False
LOCAL_N_WORKERS = 2
LOCAL_THREADS = 4
# Setting the file directories
prem_dir = "data/data/"
out_dir = "output/"
parq_dir = out_dir + "parquet/"
pkl_dir = out_dir + "pkl/"

_ = [os.makedirs(dirs, exist_ok=True) for dirs in [parq_dir, pkl_dir]]


def main():
    # %% Lazily Importing the parquet files
    print("")
    print("Loading the parquet files...")

    # Bokeh dashboard at localhost:8787
    if not ON_LIZA:
        clust = LocalCluster(
            n_workers=LOCAL_N_WORKERS, threads_per_worker=LOCAL_THREADS
        )
    else:
        # HACK: I ran this a few times and this seemed to be the sweet spot.
        clust = LocalCluster(n_workers=10, threads_per_worker=4)

    with Client(clust) as client:

        t1 = time.time()
        pq = tm.parquets_dask(dask_client=client, data_dir=prem_dir, agg_lvl=TIME_UNIT)

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

        t11 = time.time()

        print("Time to agg: {}".format(t11 - t1))

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

        # %% Writing the flat feature file to disk
        # NOTE: We repartition here based on memory size. That's not really needed
        # unless we were going to stick it in version control, but I'm doing it
        # anyway.

        print("Writing to disk")

        client.compute(
            agg_all.repartition(partition_size="100MB").to_parquet(
                parq_dir + "flat_features/", write_index=False
            ),
            sync=True,
        )

        t2 = time.time()

        print("Time total: {}".format(t2 - t1))


if __name__ == "__main__":
    main()
