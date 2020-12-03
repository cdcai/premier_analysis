import math
import pickle
from functools import reduce
from itertools import product

import dask.dataframe as dd
import dask.distributed as distributed
import pandas as pd

import tools.preprocessing as tp


def dask_merge_all(df_list, **kwargs):
    """
    A generalized reduced-join function for dask.dataframes (or pandas)

    """
    out = reduce(lambda x, y: x.join(y, **kwargs), df_list)
    return out


# Dask-enabled preprocessing class
class parquets_dask(object):
    """
    Dask-distributed Preprocessing class

    Notes:

    Useful for processing raw parquet data when many cores/workers are available.
    On a laptop/desktop, allows for slight performance increase over native pandas
    and Out-of-memory processing for larger operations.


    Visualize task graph at localhost:8787 after initializing distributed client.

    Args:

        dask_client (`obj`:`distributed.client`): A Dask distributed client instance

        data_dir (str): relative or absolute path to EHR parquet data.
        Currently this is in a submodule of the main repository so shouldn't need to be changed. (default: "data/data/")

        agg_lvl (str): Time aggregation level from index case.
        One of "dfi" (days from index), "hfi" (hours from index), or "mfi" (minutes from index) (default: "dfi")
    """

    # HACK:
    # Yeah, yeah. It's not great.
    # Could also pass in during init or just abstract away even more
    # but this is the extent of my py-fu
    final_names = ["vitals", "bill", "gen_lab", "proc", "diag", "lab_res"]
    feat_prefix = ["vtl", "bill", "genl", "proc", "dx", "lbrs"]
    time_cols = [
        ["observation_day_number", "observation_time_of_day"],
        ["serv_day"],
        ["collection_day_number", "collection_time_of_day"],
        ["proc_day"],
        None,
        ["spec_day_number", "spec_time_of_day"],
    ]

    text_cols = [
        "lab_test",
        "std_chg_desc",
        "lab_test_loinc_desc",
        "icd_code",
        "icd_code",
        "text",
    ]
    num_col = [
        "test_result_numeric_value", None, "numeric_value", None, None, None
    ]

    df_arg_names = ["df", "text_col", "feature_prefix", "num_col", "time_cols"]

    def __init__(self, dask_client, data_dir="data/data/", agg_lvl="dfi"):

        # Start Dask client
        self.client = dask_client
        print(self.client)

        # Specifying some columns to pull
        genlab_cols = [
            "pat_key",
            "collection_day_number",
            "collection_time_of_day",
            "numeric_value",
        ]
        vital_cols = [
            "pat_key",
            "observation_day_number",
            "observation_time_of_day",
            "test_result_numeric_value",
        ]
        bill_cols = ["std_chg_desc", "serv_day"]
        lab_res_cols = [
            "spec_day_number",
            "spec_time_of_day",
            "test",
            "observation",
        ]

        # Pulling in the visit tables
        self.pat = dd.read_parquet(data_dir + "vw_covid_pat/", index="pat_key")
        self.id = dd.read_parquet(data_dir + "vw_covid_id/", index="pat_key")

        # Pulling the lab and vitals
        genlab = dd.read_parquet(
            data_dir + "vw_covid_genlab/",
            columns=genlab_cols,
            index="lab_test_loinc_desc",
        )
        hx_genlab = dd.read_parquet(
            data_dir + "vw_covid_hx_genlab/",
            columns=genlab_cols,
            index="lab_test_loinc_desc",
        )
        lab_res = dd.read_parquet(data_dir + "vw_covid_lab_res/",
                                  columns=lab_res_cols,
                                  index="pat_key")

        hx_lab_res = dd.read_parquet(data_dir + "vw_covid_hx_lab_res/",
                                     columns=lab_res_cols,
                                     index="pat_key")
        vitals = dd.read_parquet(data_dir + "vw_covid_vitals/",
                                 columns=vital_cols,
                                 index="lab_test")
        hx_vitals = dd.read_parquet(data_dir + "vw_covid_hx_vitals/",
                                    columns=vital_cols,
                                    index="lab_test")

        # Concatenating the current and historical labs and vitals
        self._genlab = dd.concat([genlab, hx_genlab],
                                 axis=0,
                                 interleave_partitions=True)
        self._vitals = dd.concat([vitals, hx_vitals],
                                 axis=0,
                                 interleave_partitions=True)
        self._lab_res = dd.concat([lab_res, hx_lab_res],
                                  axis=0,
                                  interleave_partitions=True)

        # Pulling in the billing tables
        bill_lab = dd.read_parquet(data_dir + "vw_covid_bill_lab/",
                                   columns=bill_cols,
                                   index="pat_key")
        bill_pharm = dd.read_parquet(data_dir + "vw_covid_bill_pharm/",
                                     columns=bill_cols,
                                     index="pat_key")
        bill_oth = dd.read_parquet(data_dir + "vw_covid_bill_oth/",
                                   columns=bill_cols,
                                   index="pat_key")
        hx_bill = dd.read_parquet(data_dir + "vw_covid_hx_bill/",
                                  columns=bill_cols,
                                  index="pat_key")
        self._bill = dd.concat(
            [bill_lab, bill_pharm, bill_oth, hx_bill],
            axis=0,
            interleave_partitions=True,
        )

        # Pulling in the additional diagnosis and procedure tables
        pat_diag = dd.read_parquet(data_dir + "vw_covid_paticd_diag/",
                                   index="pat_key")
        pat_proc = dd.read_parquet(data_dir + "vw_covid_paticd_proc/",
                                   index="pat_key")
        add_diag = dd.read_parquet(data_dir + "vw_covid_additional_paticd_" +
                                   "diag/",
                                   index="pat_key")
        add_proc = dd.read_parquet(data_dir + "vw_covid_additional_paticd_" +
                                   "proc/",
                                   index="pat_key")
        self._diag = dd.concat([pat_diag, add_diag],
                               axis=0,
                               interleave_partitions=True)
        self._proc = dd.concat([pat_proc, add_proc],
                               axis=0,
                               interleave_partitions=True)

        # And any extras
        self.icd = dd.read_parquet(data_dir + "icdcode/")

        # Fixing lab_res
        self._lab_res["text"] = (self._lab_res["test"].astype(str) + " " +
                                 self._lab_res["observation"].astype(str))

        # Compute all the needed arguments for df_to_feature
        self.df_kwargs = self.compute_kwargs()

        # Save agg level information

        # Quick lookup for multiplier based on agg_lvl
        # NOTE: we will use these values to transform
        # day count to the appropriate time unit agg_lvl.
        # and also use them to transform the time (in seconds)
        # pulled from a timestamp to the appropriate time unit
        day_vals = [1, 24, 1440]
        sec_vals = [60 * 60 * 24, 60 * 60, 60]
        agg_lvls = ["dfi", "hfi", "mfi"]

        self.from_days = dict(zip(agg_lvls, day_vals))
        self.from_seconds = dict(zip(agg_lvls, sec_vals))

        self.agg_level = agg_lvl

        # Compute timing from index for each visit
        self.visit_timing = self.client.compute(self.get_visit_timing(self.id))

        # Pull id as a pandas dataframe since it's not too large
        self.id = self.client.compute(self.id)

        # Compute an H:M:S lookup table
        # (which is quick and prevents us from parsing or using string ops)
        time_stamps = [
            "{:02d}:{:02d}:{:02d}".format(a, b, c)
            for a, b, c in product(range(24), range(60), range(60))
        ]

        self.time_dict = dict(zip(time_stamps, range(len(time_stamps))))

        return

    def compute_kwargs(self):
        out = [
            dict(zip(self.df_arg_names, a)) for a in zip(
                [
                    self._vitals,
                    self._bill,
                    self._genlab,
                    self._proc,
                    self._diag,
                    self._lab_res,
                ],
                self.text_cols,
                self.feat_prefix,
                self.num_col,
                self.time_cols,
            )
        ]

        return out

    def all_df_to_feat(self, pkl_file=None):
        """
        An all-in-one method to standardize numeric and text columns into token features.

        Note:
        This process is inspired by Rajkomar et al. 2018. Numeric features (lab, vitals) are discretized
        by taking sample quantiles. Text features are taken as-is. Each feature is mapped to its source feature
        and condensed.

        Args:
            pkl_file (str): relative or absolute path pointing to where the pickled feature lookup dictionaries should be written
            (default: None)
        """

        out = []
        code_dicts = []

        # Quantize numeric output, rename text column to "text"
        # NOTE: This is persistent
        out = [self.df_to_features(**kw) for kw in self.df_kwargs]

        # Compute all feature values for each table for lookup table
        code_dicts = [
            self.col_to_features(a["text"], b["feature_prefix"])
            for a, b in zip(out, self.df_kwargs)
        ]

        # Convert long feature name to condensed value computed in code dict
        out = [
            self.condense_features(a, "text", b)
            for a, b in zip(out, code_dicts)
        ]

        # Combining the feature dicts and saving to disk
        ftr_dict = dict(
            zip(
                tp.flatten([d.keys() for d in code_dicts]),
                tp.flatten([d.values() for d in code_dicts]),
            ))

        # Write all dictionaries to pickle
        if pkl_file is not None:
            with open(pkl_file, "wb") as file:
                pickle.dump(ftr_dict, file)

        # return dict of dask dataframes which contain the slimmed down data
        # NOTE: These still have to be evaluated, but
        # the hope is keeping it as a task graph will keep memory overhead low
        # until it's absolutely necessary to read in
        return dict(zip(self.final_names, out)), ftr_dict

    def condense_features(self, df, text_col="text", code_dict=None):
        """Map long-named features to condensed names"""
        df["ftr"] = df[text_col].map({k: v for v, k in code_dict.items()})

        df = df.drop(text_col, axis=1)

        return self.client.persist(df)

    def col_to_features(self, text, feature_prefix):
        """Create dictionary of feature token names"""
        unique_codes = self.client.compute(text.unique(), sync=True)
        n_codes = len(unique_codes)
        ftr_codes = ["".join([feature_prefix, str(i)]) for i in range(n_codes)]
        code_dict = dict(zip(ftr_codes, unique_codes))

        return code_dict

    @staticmethod
    def _compute_approx_quantiles(
        df: dd.DataFrame,
        text_col: str,
        num_col: str,
        time_col: str,
        cuts=[0, 0.25, 0.5, 0.75, 1],
    ) -> dict:
        """
        Compute t-digest streaming approximate quantiles on large data
        in order to bin them
        """
        pivoted = df.categorize(text_col).pivot_table(index=time_col,
                                                      columns=text_col,
                                                      values=num_col)

        out = pivoted.quantile(q=cuts, method="tdigest").compute()

        return out.to_dict("list")

    def df_to_features(
        self,
        df: dd.DataFrame,
        text_col: str,
        feature_prefix: str,
        num_col=None,
        time_cols=None,
        buckets=5,
        slim=True,
    ):
        """Transform raw text and numeric features to token feature columns"""

        # Optionally quantizing the numeric column
        if num_col is not None:

            df["q"] = (df.groupby(text_col)[num_col].apply(lambda x: pd.qcut(
                x, q=buckets, labels=False, duplicates="drop")).reset_index(
                    drop=True))
            # NOTE: Vitals and genlab come in indexed by text_col, but we need to reindex by pat_key
            df = df.reset_index().set_index("pat_key").persist()

            df[text_col] = df[text_col].astype(str)

            df[text_col] += " q" + df["q"].astype(str)
        else:
            df[text_col] = df[text_col].astype(str)
        # Return full set (as pandas DF)
        if not slim:
            return df

        # Rename text_col to "text" for further downstream processing
        df = df.rename(columns={text_col: "text"})

        out_cols = ["text", "q"]

        if time_cols is not None:
            out_cols += time_cols

        # Return as a dask lazy Df and a persistent dict with the features
        out = df[out_cols]

        return self.client.persist(out)

    def get_visit_timing(self, id_table, day_col="days_from_index"):
        """Convert starting visit times to appropriate time units"""
        out = id_table[day_col].to_frame()

        out[self.agg_level] = out[day_col] * self.from_days[self.agg_level]

        out = out.drop(day_col, axis=1)

        return out

    def get_timing(self,
                   df,
                   day_col=None,
                   end_of_visit=False,
                   time_col=None,
                   ftr_col="ftr"):
        """Compute timing for each feature based on the granularity specified.

        Arguments:

        df: Dask or pandas dataframe
            Data where time and feature columns can be found

        day_col: str (default: None)
            Name of column in df which contains days from record index

        end_of_visit: bool (default: False)
            Should the feature values be appended to the last day in the visit?

        time_col: str (default: None)
            Optional column name in df which contains intra-day timing

        ftr_col: str (default: "ftr")
            Name of feature column to aggregate
        """

        # Compute which cols we will have
        out_cols = [
            col for col in [ftr_col, day_col, time_col] if col is not None
        ]

        # Merge in timing for visit which was already computed
        # make sure to index by pat_key so we can keep this efficient.
        # out: dask df with [out_cols], agg_lvl
        out = df[out_cols]
        out = out.join(self.visit_timing.result(), how="left", on="pat_key")

        # If we want this to occur at the end of the visit, add visit LOS to our timing
        # that was computed in get_visit_timing
        if end_of_visit:
            out = out.join(self.id.result()["los"].to_frame(),
                           how="left",
                           on="pat_key")

            out[self.agg_level] += out["los"] * self.from_days[self.agg_level]
            out = out.drop("los", axis=1)

        # If we have no other timing information, the visit timing is
        # all we can add, so return as-is
        if day_col is None:
            return out

        # Add day column contribution to the timing
        out[self.agg_level] += out[day_col] * self.from_days[self.agg_level]

        # We don't need the day col anymore
        out = out.drop(day_col, axis=1)

        if time_col is not None:
            # Using a seconds in a day dictionary lookup
            # map HMS strings to dict to convert to seconds
            # then aggregate to appropriate agg_level

            out[self.agg_level] += (out[time_col].map(self.time_dict) /
                                    self.from_seconds[self.agg_level])

            # Done with the time col
            out = out.drop(time_col, axis=1)

        return out

    def reset_agg_level(self, new_level):
        """Helper function to reset aggregation level during runtime
        in the unlikely event that you'd like to recompute at a different
        level of time aggregation without re-running the whole pipeline.
        """
        # Update agg level
        self.agg_level = new_level

        # Recompute visit timing based on new agg level
        self.visit_timing = self.get_visit_timing(self.id)

        return None

    def agg_features(self, df, as_str=True, ftr_col="ftr", out_col="ftrs"):
        """
        Aggregate feature column to token columns by time step + id

        Notes:

        This defaults to joining all tokens present at the given time step to a sting, but can also
        be left as a list instead of string for easier downstream processing.

        Args:
            df (`obj`:`dask.DataFrame` or `pandas.DataFrame`): The source DataFrame after it's been run through df_to_features.
            Can be persistent or Dask.

            as_str (`bool`): Should the resulting feature be a string or left as a list (default: True)

            ftr_col (`str`): Name of column where features to aggregate can be found (default: "ftr")

            out_col (`str`): Name of resulting column (default: "ftrs")
        """

        grouped = df[[self.agg_level,
                      ftr_col]].groupby([self.agg_level, df.index])
        agged = grouped.agg(list).rename(columns={ftr_col: out_col})

        # We might want to keep as list-of-lists instead of concatenating
        if as_str:
            agged[out_col] = agged[out_col].map(lambda x: " ".join(x))

        return self.client.persist(agged)
