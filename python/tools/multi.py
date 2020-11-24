"""
Multiprocessing-enabled versions of functions from tools.py
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy.stats import chi2, norm
from copy import deepcopy
from multiprocessing import Pool

import tools.preprocessing as tp


def get_times(df, dict, day_col=None, time_col=None, ftr_col="ftr"):
    """Gets days, hours, and minutes from index for a table"""
    # Doing the days
    dfi_orig = np.array([dict[id] for id in df.pat_key])

    # Optionally returning early if the table has no existing day or time
    if day_col is None:
        out = df[["pat_key", ftr_col]]
        out["dfi"] = dfi_orig
        out["hfi"] = out.dfi * 24
        out["mfi"] = out.hfi * 60
        return out

    # Doing the hours and minutes
    dfi = np.array(dfi_orig + df[day_col], dtype=np.uint32)
    hfi = dfi * 24
    mfi = hfi * 60
    if time_col is not None:
        p = Pool()
        times = [t for t in df[time_col]]
        hours = np.array(p.map(tp.time_to_hours, times), dtype=np.uint32)
        mins = np.array(p.map(tp.time_to_minutes, times), dtype=np.uint32)
        p.close()
        p.join()
        hfi += hours
        mfi += mins

    # Returning the new df
    out = df[["pat_key", ftr_col]]
    out["dfi"] = dfi
    out["hfi"] = hfi
    out["mfi"] = mfi

    return out


def jackknife_metrics(targets, guesses, average_by=None, weighted=True):
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
    def __init__(
        self,
        targets,
        guesses,
        sample_by=None,
        n=100,
        a=0.05,
        method="bca",
        interpolation="nearest",
        average_by=None,
        weighted=True,
        mcnemar=False,
        seed=10221983,
    ):
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
            inputs = [(targets[boot], guesses[boot], average_by[boot],
                       weighted) for boot in boots]
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
        methods = ["pct", "diff", "bca"]
        assert method in methods, "Method must be pct, diff, or bca."

        # Calculating the CIs with method #1: the percentiles of the
        # bootstrapped statistics
        if method == "pct":
            cis = np.nanpercentile(scores,
                                   q=(lower, upper),
                                   interpolation=interpolation,
                                   axis=0)
            cis = pd.DataFrame(cis.transpose(),
                               columns=["lower", "upper"],
                               index=colnames)

        # Or with method #2: the percentiles of the difference between the
        # obesrved statistics and the bootstrapped statistics
        elif method == "diff":
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
        elif method == "bca":
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
            denom = 6 * np.power(np.sum(np.power(diffs, 2)), 3 / 2)

            # Getting rid of 0s in the denominator
            zeros = np.where(denom == 0)[0]
            for z in zeros:
                denom[z] += 1e-6

            # Finishing up the acceleration parameter
            acc = numer / denom
            self.jack = j

            # Calculating the bounds for the confidence intervals
            zl = norm.ppf(a / 2)
            zu = norm.ppf(1 - (a / 2))
            lterm = (z0 + zl) / (1 - acc * (z0 + zl))
            uterm = (z0 + zu) / (1 - acc * (z0 + zu))
            lower_q = norm.cdf(z0 + lterm) * 100
            upper_q = norm.cdf(z0 + uterm) * 100
            self.lower_q = lower_q
            self.upper_q = upper_q

            # Returning the CIs based on the adjusted quintiles
            cis = [
                np.nanpercentile(
                    scores.iloc[:, i],
                    q=(lower_q[i], upper_q[i]),
                    interpolation=interpolation,
                    axis=0,
                ) for i in range(len(lower_q))
            ]
            cis = pd.DataFrame(cis, columns=["lower", "upper"], index=colnames)

        # Putting the stats with the lower and upper estimates
        cis = pd.concat([stat, cis], axis=1)
        cis.columns = ["stat", "lower", "upper"]

        # Passing the results back up to the class
        self.cis = cis
        self.scores = scores

        return


def boot_roc(targets, scores, sample_by=None, n=1000, seed=10221983):
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
