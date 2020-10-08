import numpy as np
import pandas as pd
import multiprocessing
import os
import sys

from functools import reduce
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder


# Turning off the pandas chained assignment warning
pd.options.mode.chained_assignment = None


class load_parquets:
    def __init__(self,
                 dir='data/data/'):
        # Specifying some columns to pull
        genlab_cols = ['pat_key', 'collection_day_number',
                       'collection_time_of_day', 'lab_test_loinc_desc',
                       'numeric_value']
        vital_cols = ['pat_key', 'observation_day_number',
                      'observation_time_of_day', 'lab_test',
                      'test_result_numeric_value']
        bill_cols = ['pat_key', 'std_chg_desc', 'serv_day']
        lab_res_cols = ['pat_key', 'spec_day_number', 'spec_time_of_day',
                        'test', 'observation']
        
        # Pulling in the visit tables
        self.pat = pd.read_parquet(dir + 'vw_covid_pat/')
        self.id = pd.read_parquet(dir + 'vw_covid_id/')
        
        # Pulling the lab and vitals
        genlab = pd.read_parquet(dir + 'vw_covid_genlab/',
                                      columns=genlab_cols)
        hx_genlab = pd.read_parquet(dir + 'vw_covid_hx_genlab/',
                                         columns=genlab_cols)
        lab_res = pd.read_parquet(dir + 'vw_covid_lab_res/',
                                       columns=lab_res_cols)
        hx_lab_res = pd.read_parquet(dir + 'vw_covid_hx_lab_res/',
                                          columns=lab_res_cols)
        vitals = pd.read_parquet(dir + 'vw_covid_vitals/',
                                      columns=vital_cols)
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
        hx_bill = pd.read_parquet(dir + 'vw_covid_hx_bill/',
                                       columns=bill_cols)
        self.bill = pd.concat([bill_lab, bill_pharm, 
                               bill_oth, hx_bill], axis=0)
        
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
                   time_cols=None,
                   buckets=5,
                   slim=True):
    # Pulling out the text
    text = df[text_col].astype(str)
    
    # Optionally quantizing the numeric column
    if num_col is not None:
        df['q'] = df.groupby(text_col)[num_col].transform(
                     lambda x: pd.qcut(x, 
                                       buckets,
                                       labels=False, 
                                       duplicates='drop')
                     )
        text += ' q' + df.q.astype(str)
    
    # Making a lookup dict for the features
    ftr_dict, code_dict = col_to_features(text, feature_prefix)
    
    # Adding the combined feature col back into the original df
    ftr = [ftr_dict[code] for code in text]
    
    # Combining the features with the original data
    df['ftr'] = ftr
    
    # Optionally slimming down the output data frame
    if slim:
        out_cols = ['pat_key', 'ftr']
        if time_cols is not None:
            out_cols += time_cols
        return df[out_cols], code_dict
    else:
        return df, code_dict


def agg_features(df, 
                 time_col=None, 
                 id_col='pat_key', 
                 ft_col='ftr'):
    '''Aggregates feature tokens by time.'''
    if time_col is not None:
        grouped = df.groupby([id_col, time_col], as_index=False)
        agged = grouped[ft_col].agg({'ftrs': ' '.join})
        agged.columns = [id_col, time_col, 'ftrs']
        agged[time_col] = agged[time_col].astype(int)
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
    return h*60 + m


def time_to_hours(s):
    '''Converts a hh:mm:ss string to integer hours'''
    if s is None:
        return 0
    return np.round(time_to_minutes(s) / 60).astype(int)
    

def merge_all(df_list, on='pat_key', how='outer'):
    out = reduce(lambda x, y: pd.merge(x, y, on=on, how=how), df_list)
    return out


def sparsify(col, 
             reshape=True, 
             return_df=True,
             long_names=False):
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


    