'''
Multiprocessing-enabled versions of functions from tools.py
'''

import pandas as pd
import numpy as np
import pickle

import dask.dataframe as dd
from dask import delayed
from dask.distributed import Client

from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy.stats import chi2, norm
from copy import deepcopy
from multiprocessing import Pool

import tools.preprocessing as tp


def get_times(df,
              dict, 
              day_col=None, 
              time_col=None,
              ftr_col='ftr'):
    '''Gets days, hours, and minutes from index for a table'''
    # Doing the days
    dfi_orig = np.array([dict[id] for id in df.pat_key])
    
    # Optionally returning early if the table has no existing day or time
    if day_col is None:
        out = df[['pat_key', ftr_col]]
        out['dfi'] = dfi_orig
        out['hfi'] = out.dfi * 24
        out['mfi'] = out.hfi * 60
        return out
    
    # Doing the hours and minutes
    dfi = np.array(dfi_orig + df[day_col], dtype=np.uint32)
    hfi = dfi * 24
    mfi = hfi * 60
    if time_col is not None:
        p = Pool()
        times = [t for t in df[time_col]]
        hours = np.array(p.map(tp.time_to_hours, times),
                         dtype=np.uint32)
        mins = np.array(p.map(tp.time_to_minutes, times),
                        dtype=np.uint32)
        p.close()
        p.join()
        hfi += hours
        mfi += mins
    
    # Returning the new df
    out = df[['pat_key', ftr_col]]
    out['dfi'] = dfi
    out['hfi'] = hfi
    out['mfi'] = mfi
        
    return out


def jackknife_metrics(targets,
                      guesses,
                      average_by=None,
                      weighted=True):
        # Replicates of the dataset with one row missing from each
        rows = np.array(list(range(targets.shape[0])))
        j_rows = [np.delete(rows, row) for row in rows]
        
        # using a pool to get the metrics across each
        inputs = [(targets[idx], guesses[idx], average_by, weighted)
                  for idx in j_rows]
        p = Pool()
        stat_list = p.starmap(tools.clf_metrics, inputs)
        p.close()
        p.join()
        
        # Combining the jackknife metrics and getting their means
        scores = pd.concat(stat_list, axis=0)
        means = scores.mean()
        return scores, means


# Calculates bootstrap confidence intervals for an estimator
class boot_cis:
    def __init__(self,
                 targets, 
                 guesses,
                 sample_by=None,
                 n=100,
                 a=0.05,
                 method='bca', 
                 interpolation='nearest',
                 average_by=None,
                 weighted=True,
                 mcnemar=False,
                 seed=10221983):
        # Converting everything to NumPy arrays, just in case
        stype = type(pd.Series())
        if type(sample_by) == stype:
            sample_by = sample_by.values
        if type(targets) == stype:
            targets = targets.values
        if type(guesses) == stype:
            guesses = guesses.values
        
        # Getting the point estimates
        stat = tools.clf_metrics(targets,
                                 guesses,
                                 average_by=average_by,
                                 weighted=weighted,
                                 mcnemar=mcnemar).transpose()
        
        # Pulling out the column names to pass to the bootstrap dataframes
        colnames = list(stat.index.values)
        
        # Making an empty holder for the output
        scores = pd.DataFrame(np.zeros(shape=(n, stat.shape[0])),
                              columns=colnames)
        
        # Setting the seed
        if seed is None:
            seed = np.random.randint(0, 1e6, 1)
        np.random.seed(seed)
        seeds = np.random.randint(0, 1e6, n)
        
        # Generating the bootstrap samples and metrics
        p = Pool()
        boot_input = [(targets, sample_by, None, seed) for seed in seeds]
        boots = p.starmap(tools.boot_sample, boot_input)
        
        if average_by is not None:
            inputs = [(targets[boot],
                       guesses[boot],
                       average_by[boot],
                       weighted)
                      for boot in boots]
        else:
            inputs = [(targets[boot], guesses[boot]) for boot in boots]
        
        # Getting the bootstrapped metrics from the Pool
        p_output = p.starmap(tools.clf_metrics, inputs)
        scores = pd.concat(p_output, axis=0)
        p.close()
        p.join()
        
        # Calculating the confidence intervals
        lower = (a / 2) * 100
        upper = 100 - lower
        
        # Making sure a valid method was chosen
        methods = ['pct', 'diff', 'bca']
        assert method in methods, 'Method must be pct, diff, or bca.'
        
        # Calculating the CIs with method #1: the percentiles of the 
        # bootstrapped statistics
        if method == 'pct':
            cis = np.nanpercentile(scores, 
                                   q=(lower, upper),
                                   interpolation=interpolation, 
                                   axis=0)
            cis = pd.DataFrame(cis.transpose(),
                               columns=['lower', 'upper'],
                               index=colnames)
        
        # Or with method #2: the percentiles of the difference between the
        # obesrved statistics and the bootstrapped statistics
        elif method == 'diff':
            stat_vals = stat.transpose().values.ravel()
            diffs = stat_vals - scores
            percents = np.nanpercentile(diffs,
                                        q=(lower, upper),
                                        interpolation=interpolation,
                                        axis=0)
            lower_bound = pd.Series(stat_vals + percents[0])
            upper_bound = pd.Series(stat_vals + percents[1])
            cis = pd.concat([lower_bound, upper_bound], axis=1)
            cis = cis.set_index(stat.index)
        
        # Or with method #3: the bias-corrected and accelerated bootstrap
        elif method == 'bca':
            # Calculating the bias-correction factor
            stat_vals = stat.transpose().values.ravel()
            n_less = np.sum(scores < stat_vals, axis=0)
            p_less = n_less / n
            z0 = norm.ppf(p_less)
            
            # Fixing infs in z0
            z0[np.where(np.isinf(z0))[0]] = 0.0
            
            # Estiamating the acceleration factor
            j = jackknife_metrics(targets, guesses)
            diffs = j[1] - j[0]
            numer = np.sum(np.power(diffs, 3))
            denom = 6 * np.power(np.sum(np.power(diffs, 2)), 3/2)
            
            # Getting rid of 0s in the denominator
            zeros = np.where(denom == 0)[0]
            for z in zeros:
                denom[z] += 1e-6
            
            # Finishing up the acceleration parameter
            acc = numer / denom
            self.jack = j
            
            # Calculating the bounds for the confidence intervals
            zl = norm.ppf(a / 2)
            zu = norm.ppf(1 - (a/2))
            lterm = (z0 + zl) / (1 - acc*(z0 + zl))
            uterm = (z0 + zu) / (1 - acc*(z0 + zu))
            lower_q = norm.cdf(z0 + lterm) * 100
            upper_q = norm.cdf(z0 + uterm) * 100
            self.lower_q = lower_q
            self.upper_q = upper_q
            
            # Returning the CIs based on the adjusted quintiles
            cis = [np.nanpercentile(scores.iloc[:, i], 
                                    q=(lower_q[i], upper_q[i]),
                                    interpolation=interpolation, 
                                    axis=0) 
                   for i in range(len(lower_q))]
            cis = pd.DataFrame(cis,
                               columns=['lower', 'upper'],
                               index=colnames)
        
        # Putting the stats with the lower and upper estimates
        cis = pd.concat([stat, cis], axis=1)
        cis.columns = ['stat', 'lower', 'upper']
        
        # Passing the results back up to the class
        self.cis = cis
        self.scores = scores
        
        return


def boot_roc(targets,
             scores,
             sample_by=None,
             n=1000,
             seed=10221983):
    # Generating the seeds
    np.random.seed(seed)
    seeds = np.random.randint(1, 1e7, n)
    
    # Getting the indices for the bootstrap samples
    p = Pool()
    boot_input = [(targets, sample_by, None, seed) for seed in seeds]
    boots = p.starmap(tools.boot_sample, boot_input)
    
    # Getting the ROC curves
    roc_input = [(targets[boot], scores[boot]) for boot in boots]
    rocs = p.starmap(roc_curve, roc_input)
    
    return rocs

# Dask-enabled preprocessing class
class load_parquets_dask(object):

    # HACK:
    # Yeah, yeah. It's not great.
    # Could also pass in during init or just abstract away even more
    # but this is the extent of my py-fu
    final_names = ['vitals', 'bill', 'gen_lab', 'proc', 'diag', 'lab_res']
    dask_frames = ['_vitals', '_bill', '_genlab', '_proc', '_diag', '_lab_res']
    feat_prefix = ['vtl', 'bill', 'genl', 'proc', 'dx', 'lbrs']
    time_cols = [
        ['observation_day_number','observation_time_of_day'],
        ['serv_day'],
        ['collection_day_number', 'collection_time_of_day'],
        ['proc_day'],
        None,
        ['spec_day_number', 'spec_time_of_day']
    ]

    text_cols = ['lab_test', 'std_chg_desc', 'lab_test_loinc_desc', 'icd_code', 'icd_code', 'text']
    num_col = ['test_result_numeric_value', None, 'numeric_value', None, None, None]

    df_arg_names = ['df', 'text_col', 'feature_prefix', 'num_col', 'time_cols']

    def __init__(self, dir='data/data/', pkl_dict='../output/pkl/feature_lookup.pkl'):

        # Start Dask client
        self.client = Client(processes=False)
        print(self.client)

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
        self.pat = dd.read_parquet(dir + 'vw_covid_pat/')
        self.id = dd.read_parquet(dir + 'vw_covid_id/')
        
        # Pulling the lab and vitals
        genlab = dd.read_parquet(dir + 'vw_covid_genlab/',
                                      columns=genlab_cols)
        hx_genlab = dd.read_parquet(dir + 'vw_covid_hx_genlab/',
                                         columns=genlab_cols)
        lab_res = dd.read_parquet(dir + 'vw_covid_lab_res/',
                                       columns=lab_res_cols)
        
        hx_lab_res = dd.read_parquet(dir + 'vw_covid_hx_lab_res/',
                                          columns=lab_res_cols)
        vitals = dd.read_parquet(dir + 'vw_covid_vitals/',
                                      columns=vital_cols)
        hx_vitals = dd.read_parquet(dir + 'vw_covid_hx_vitals/',
                                         columns=vital_cols)
        
        # Concatenating the current and historical labs and vitals
        self._genlab = dd.concat([genlab, hx_genlab], axis=0, interleave_partitions=True)
        self._vitals = dd.concat([vitals, hx_vitals], axis=0, interleave_partitions=True)
        self._lab_res = dd.concat([lab_res, hx_lab_res], axis=0, interleave_partitions=True)
        
        # Pulling in the billing tables
        bill_lab = dd.read_parquet(dir + 'vw_covid_bill_lab/',
                                   columns=bill_cols)
        bill_pharm = dd.read_parquet(dir + 'vw_covid_bill_pharm/',
                                     columns=bill_cols)
        bill_oth = dd.read_parquet(dir + 'vw_covid_bill_oth/',
                                   columns=bill_cols)
        hx_bill = dd.read_parquet(dir + 'vw_covid_hx_bill/',
                                       columns=bill_cols)
        self._bill = dd.concat([bill_lab, bill_pharm, 
                               bill_oth, hx_bill], axis=0, interleave_partitions=True)
        
        # Pulling in the additional diagnosis and procedure tables
        pat_diag = dd.read_parquet(dir + 'vw_covid_paticd_diag/')
        pat_proc = dd.read_parquet(dir + 'vw_covid_paticd_proc/')
        add_diag = dd.read_parquet(dir + 'vw_covid_additional_paticd_' +
                                        'diag/')
        add_proc = dd.read_parquet(dir + 'vw_covid_additional_paticd_' +
                                        'proc/')
        self._diag = dd.concat([pat_diag, add_diag], axis=0, interleave_partitions=True)
        self._proc = dd.concat([pat_proc, add_proc], axis=0, interleave_partitions=True)
                
        # And any extras
        self.icd = dd.read_parquet(dir + 'icdcode/')

        # Fixing lab_res
        self._lab_res['text'] = self._lab_res['test'].astype(str) + ' ' + self._lab_res['observation'].astype(str)

        # Compute all the needed arguments for df_to_feature
        self.df_kwargs = self.compute_kwargs()

        # NOTE: Creates dict of dask dataframes for each table
        # but does not read into memory without a compute() call
        # so the memory overhead is lower. Also writes out a pickle file
        # with all of the dicts
        self.data = self.all_df_to_feat(pkl_dict)

        return
    
    def compute_kwargs(self):
        out = [dict(zip(self.df_arg_names, a)) for a in zip(self.dask_frames, self.text_cols, self.feat_prefix, self.num_col, self.time_cols)]

        return out

    def all_df_to_feat(self, outfile):

        out_promise = []

        for kwargs in self.df_kwargs:
            out_promise.append(self.client.compute(self.df_to_features(**kwargs), sync=False))
        
        # BUG: This is pretty neophyte-level parallelism,
        # revisit eventually if you ever think of a better approach
        out = [promise.result() for promise in out_promise]

        # Combining the feature dicts and saving to disk
        ftr_dict = dict(zip(tp.flatten([d.keys() for _, d in out]),
                            tp.flatten([d.values() for _, d in out])))

        # Write all dictionaries to pickle
        with open(outfile, 'wb') as file:
            pickle.dump(ftr_dict, file)
        
        # return dict of dask dataframes which contain the slimmed down data
        # NOTE: These still have to be evaluated, but
        # the hope is keeping it as a task graph will keep memory overhead low
        # until it's absolutely necessary to read in
        return dict(zip(self.final_names, [a for a, _ in out]))

    def col_to_features(self, text, feature_prefix):

        unique_codes = self.client.compute(text.unique(), sync = True)
        n_codes = len(unique_codes) 
        ftr_codes = [''.join([feature_prefix, str(i)]) for i in range(n_codes)]
        code_dict = dict(zip(ftr_codes, unique_codes))

        return code_dict

    def num_to_quant(self, df, text, num, buckets):
        df = df.set_index(text)
        
        df['q'] = (df
            .groupby(text)[num]
            .transform(pd.qcut, q=buckets, labels=False, duplicates='drop')
        )

        df = df.reset_index(drop=False)

        df[text] += ' q' + df['q'].astype(str)

        return df

    @delayed
    def df_to_features(self, df,
                   text_col,
                   feature_prefix,
                   num_col=None,
                   time_cols=None,
                   buckets=5,
                   slim=True):

        df_local = getattr(self, df)

        # Pulling out the text
        df_local[text_col] = df_local[text_col].astype(str)
        
        # Optionally quantizing the numeric column
        if num_col is not None:

            df_local = self.num_to_quant(df_local, text_col, num_col, buckets)
        
        # text = self.client.compute(df_local[text_col], sync=True)
        
        # Making a lookup dict for the features
        code_dict = self.col_to_features(df_local[text_col], feature_prefix)
        
        # Return full set (as pandas DF)
        if not slim:
            return client.compute(df_local, sync=True), code_dict
        
        out_cols = ['pat_key']

        if time_cols is not None:
            out_cols += time_cols
        
        out = df_local[out_cols]

        # Adding the combined feature col back into the original df
        out['ftr'] = df_local[text_col].map({k: v for v, k in code_dict.items()})

        # Return as a dask lazy Df and a persistent dict with the features
        return out, code_dict

