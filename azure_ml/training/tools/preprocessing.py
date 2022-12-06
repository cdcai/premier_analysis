'''Classes and functions to support feature_extraction.py'''

import numpy as np
import pandas as pd
import os
import sys

from functools import reduce
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple


# Turning off the pandas chained assignment warning
pd.options.mode.chained_assignment = None


class load_parquets:
    def __init__(self, dir='data/data/'):
        # Pulling the absolute path to avoid weirdness with pd.read_parquet
        dir = os.path.abspath(dir) + '/'

        # Specifying some columns to pull
        genlab_cols = [
            'pat_key', 'collection_day_number', 'collection_time_of_day',
            'lab_test_loinc_desc', 'lab_test_result', 'numeric_value'
        ]
        vital_cols = [
            'pat_key', 'observation_day_number', 'observation_time_of_day',
            'lab_test', 'test_result_numeric_value'
        ]
        bill_cols = ['pat_key', 'std_chg_desc', 'serv_day']
        lab_res_cols = [
            'pat_key', 'spec_day_number', 'spec_time_of_day', 'test',
            'observation'
        ]

        # Pulling in the visit tables
        self.pat = pd.read_parquet(dir + 'vw_covid_pat/')
        self.id = pd.read_parquet(dir + 'vw_covid_id/')

        # Pulling the lab and vitals
        genlab = pd.read_parquet(dir + 'vw_covid_genlab/', columns=genlab_cols)
        hx_genlab = pd.read_parquet(dir + 'vw_covid_hx_genlab/',
                                    columns=genlab_cols)
        lab_res = pd.read_parquet(dir + 'vw_covid_lab_res/',
                                  columns=lab_res_cols)
        hx_lab_res = pd.read_parquet(dir + 'vw_covid_hx_lab_res/',
                                     columns=lab_res_cols)
        vitals = pd.read_parquet(dir + 'vw_covid_vitals/', columns=vital_cols)
        hx_vitals = pd.read_parquet(dir + 'vw_covid_hx_vitals/',
                                    columns=vital_cols)

        # Concatenating the current and historical labs and vitals
        self.genlab = pd.concat([genlab, hx_genlab], axis=0)
        self.vitals = pd.concat([vitals, hx_vitals], axis=0)
        self.lab_res = pd.concat([lab_res, hx_lab_res], axis=0)

        # Pulling in the billing tables
        bill_lab = pd.read_parquet(dir + 'vw_covid_bill_lab/',
                                   columns=bill_cols)
        bill_pharm = pd.read_parquet(dir + 'vw_covid_bill_pharm/',
                                     columns=bill_cols)
        bill_oth = pd.read_parquet(dir + 'vw_covid_bill_oth/',
                                   columns=bill_cols)
        hx_bill = pd.read_parquet(dir + 'vw_covid_hx_bill/', columns=bill_cols)
        self.bill = pd.concat([bill_lab, bill_pharm, bill_oth, hx_bill],
                              axis=0)

        # Pulling in the additional diagnosis and procedure tables
        pat_diag = pd.read_parquet(dir + 'vw_covid_paticd_diag/')
        pat_proc = pd.read_parquet(dir + 'vw_covid_paticd_proc/')
        add_diag = pd.read_parquet(dir + 'vw_covid_additional_paticd_' +
                                   'diag/')
        add_proc = pd.read_parquet(dir + 'vw_covid_additional_paticd_' +
                                   'proc/')
        self.diag = pd.concat([pat_diag, add_diag], axis=0)
        self.proc = pd.concat([pat_proc, add_proc], axis=0)

        # Resetting indexes for the concatenated tables
        self.bill.reset_index(drop=True, inplace=True)
        self.genlab.reset_index(drop=True, inplace=True)
        self.vitals.reset_index(drop=True, inplace=True)
        self.lab_res.reset_index(drop=True, inplace=True)
        self.diag.reset_index(drop=True, inplace=True)
        self.proc.reset_index(drop=True, inplace=True)

        # And any extras
        self.icd = pd.read_parquet(dir + 'icdcode/')

        return


def col_to_features(text, feature_prefix):
    unique_codes = text.unique()
    n_codes = len(unique_codes)
    ftr_codes = [''.join([feature_prefix, str(i)]) for i in range(n_codes)]
    ftr_dict = dict(zip(unique_codes, ftr_codes))
    code_dict = dict(zip(ftr_codes, unique_codes))
    return ftr_dict, code_dict


def df_to_features(df,
                   text_col,
                   feature_prefix,
                   num_col=None,
                   replace_col=None,
                   time_cols=None,
                   buckets=5,
                   slim=True):
    # Pulling out the text
    text = df[text_col].astype(str)
    df[text_col] = text

    # Optionally quantizing the numeric column
    if num_col is not None:
        # Converting the numeric values to quantiles
        df['q'] = df.groupby(text_col)[num_col].transform(lambda x: pd.qcut(x=x, q=buckets, labels=False, duplicates='drop'))

        # Figuring out which tests have non-numeric results
        missing_num = np.where(np.isnan(df.q))[0]

        # Converting the quantiles to strings
        qstr = [doc for doc in ' q' + df.q.astype(str)]

        # Replacing missing numerics with the original test result
        if replace_col is not None:
            for i in missing_num:
                rep = df[replace_col][i]
                if rep is not None:
                    qstr[i] = ' ' + rep
                else:
                    qstr[i] = ' none'

        # Adding the quantiles back to the text column
        text += pd.Series(qstr)

    # Making a lookup dict for the features
    ftr_dict, code_dict = col_to_features(text, feature_prefix)

    # Adding the combined feature col back into the original df
    ftr = [ftr_dict[code] for code in text]

    # Combining the features with the original data
    df['ftr'] = pd.Series(ftr, dtype=str)

    # Optionally slimming down the output data frame
    if slim:
        out_cols = ['pat_key', 'ftr']
        if time_cols is not None:
            out_cols += time_cols
        return df[out_cols], code_dict
    else:
        return df, code_dict


def agg_features(df, time_col=None, id_col='pat_key', ft_col='ftr'):
    '''Aggregates feature tokens by time.'''
    # Makign sure the features are strings
    df[ft_col] = df[ft_col].astype(str)

    # Aggregating by time if time is provided
    if time_col is not None:
        grouped = df.groupby([id_col, time_col], as_index=False)
        agged = grouped[ft_col].agg({'ftrs': ' '.join})
        agged.columns = [id_col, time_col, 'ftrs']
        agged[time_col] = agged[time_col].astype(int)

    # Otherwise aggregating by ID
    else:
        grouped = df.groupby(id_col, as_index=False)
        agged = grouped[ft_col].agg({'ftrs': ' '.join})

    return agged


def time_to_minutes(s):
    '''Converts a hh:mm:ss string to integer minutes'''
    if s is None:
        return 0
    h = int(s[0:2])
    m = int(s[3:5])
    return h * 60 + m


def time_to_hours(s):
    '''Converts a hh:mm:ss string to integer hours'''
    if s is None:
        return 0
    return np.round(time_to_minutes(s) / 60).astype(int)


def merge_all(df_list, on='pat_key', how='outer'):
    out = reduce(lambda x, y: pd.merge(x, y, on=on, how=how), df_list)
    return out


def sparsify(col, reshape=True, return_df=True, long_names=False):
    '''Makes a sparse array of a data frame of categorical variables'''
    levels = np.unique(col)
    out = np.array([col == level for level in levels],
                   dtype=np.uint8).transpose()
    if long_names:
        var = col.name + '.'
        levels = [var + level for level in levels]
    columns = [col.lower() for col in levels]
    if return_df:
        out = pd.DataFrame(out, columns=columns)
    return out


def flatten(l):
    return [item for sublist in l for item in sublist]


def find_cutpoints(visit_type: list,
                   visit_length: list,
                   tail: int = 1,
                   origin: int = 0,
                   how: str = 'first') -> Tuple[Tuple[int, int], int]:
    """
    Find appropriate cutpoints in sequence of hospitalization data.

    Transform pat_data dictionary produced in features_to_integers.py to a
    tuple of cutpoints to appropriately adjust the horizon and end of lookback period
    prior to modelling.

    Args:

        visit_type (list): a list of length n_medrec with integers indicating which associated visits were COVID related
        visit_length (list): A list of length n_medrec with sublists containing the LOS for each associated visit
        tail (int): How many days into the COVID visit to be considered should be included in the lookback period? (default: 1)
        origin (int): When should the lookback start? 
            origin = 0 takes all records in domain [0, visit + `tail`]
            origin > 0 takes records in domain [maxima(visit + `tail` - origin, 0), visit + `tail`]
            Ignored if how="both", where origin starts at start of first COVID visit
        how (str): one of "first", "last", or "both" describing the COVID visit of interest (default: "first")
            how = "first": Predict on first COVID visit (currently how MISA is being defined)
            how = "last": Predict on latest COVID visit to occur
            how = "both": Predict on latest COVID visit (lookback to first COVID visit)
    
    Returns:

        Indices to pass to trim_sequence to trim

        (int, int), int
        (lookback_start, lookback_end), label_idx
    """
    assert origin >= 0, "Origin should be a positive integer or zero"

    covid_idx = np.where(np.array(visit_type) == 1)[0]
    first = np.min(covid_idx)
    first_end = np.sum(visit_length[0:first], dtype=np.uint16) + tail
    last = np.max(covid_idx)
    last_end = np.sum(visit_length[0:last], dtype=np.uint16) + tail

    if how == 'first':
        if origin != 0:
            origin = np.maximum(0, first_end - origin)
        return (origin, first_end), first
    elif how == 'last':
        if origin != 0:
            origin = np.maximum(0, last_end - origin)
        return (origin, last_end), last
    elif how == 'both':
        return (first_end, last_end), last


def trim_sequence(inputs, labels, cuts):
    """
    Provided a list of input data, labels, and cutpoints,
    trim sequences and append labels appropriately.

    Args:

        inputs (tuple): A tuple of length two where the first element is a list of sequences to be trimmed
            and the second contains a list of sample-level sequences or features to be passed thru
        labels (list): A list of visit-level labels which will be indexed by the final cut to determine
            appropriate sample label 
        cuts (list): A list of start and end indices for the list of sequences and label idx for the sample
            to use. Computed in find_cutpoints
    """
    in_start, in_end = cuts[0][0], cuts[0][1]
    label_id = cuts[1]
    return inputs[0][in_start:in_end], inputs[1], labels[label_id]

def max_age_bins(df, id_var="medrec_key", bins=np.arange(0, 111, 10)):

    # Make nice text labels
    age_cut_labs = [
        "{}-{}".format(bins[i], bins[i + 1]) for i in range(bins.size - 1)
    ]

    # Since age could roll over, take the max
    df["age"] = df.groupby(id_var)["age"].max()

    # Bin the ages and append labels
    df["age"] = np.digitize(df["age"], bins)
    df["age"] = df["age"].map(lambda x: age_cut_labs[x - 1])

    return df
