"""
Multiprocessing-enabled versions of functions from tools.py
"""

import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve
from scipy.stats import norm
from multiprocessing import Pool

import tools.preprocessing as tp
import tools.analysis as ta


def get_times(df,
              dict,
              day_col=None,
              time_col=None,
              ftr_col="ftr",
              day_only=True):
    """Gets days, hours, and minutes from index for a table"""
    # Doing the days
    dfi_orig = np.array([dict[id] for id in df.pat_key], dtype=np.uint16)

    # Optionally returning early if the table has no existing day or time
    if day_col is None:
        out = df[["pat_key", ftr_col]]
        out["dfi"] = dfi_orig
        if not day_only:
            out["hfi"] = out.dfi * 24
            out["mfi"] = out.hfi * 60
        return out

    # Doing the hours and minutes
    dfi = np.array(dfi_orig + df[day_col], dtype=np.uint32)

    # Optionally working with hours and minutes from index
    if not day_only:
        hfi = dfi * 24
        mfi = hfi * 60

        # Converting the times to hours and minutes
        if time_col is not None:
            with Pool() as p:
                times = [t for t in df[time_col]]
                hours = np.array(p.map(tp.time_to_hours, times),
                                 dtype=np.uint32)
                mins = np.array(p.map(tp.time_to_minutes, times),
                                dtype=np.uint32)
                p.close()
                p.join()
            hfi += hours
            mfi += mins

        out["hfi"] = hfi
        out["mfi"] = mfi

    # Returning the new df
    out = df[["pat_key", ftr_col]]
    out["dfi"] = dfi

    return out


def jackknife_metrics(targets, guesses, cutpoint=0.5, average="weighted"):
    # Replicates of the dataset with one row missing from each
    rows = np.array(list(range(targets.shape[0])))
    j_rows = [np.delete(rows, row) for row in rows]

    # using a pool to get the metrics across each
    inputs = [(targets[idx], guesses[idx], average, cutpoint)
              for idx in j_rows]

    with Pool() as p:
        stat_list = p.starmap(ta.clf_metrics, inputs)
        p.close()

    # Combining the jackknife metrics and getting their means
    scores = pd.concat(stat_list, axis=0)
    means = scores.mean()

    return scores, means


# Calculates bootstrap confidence intervals for an estimator
class boot_cis:
    def __init__(self,
                 targets,
                 guesses,
                 n=100,
                 a=0.05,
                 method="bca",
                 interpolation="nearest",
                 average='weighted',
                 cutpoint=0.5,
                 mcnemar=False,
                 seed=10221983):
        # Converting everything to NumPy arrays, just in case
        stype = type(pd.Series())
        if type(targets) == stype:
            targets = targets.values
        if type(guesses) == stype:
            guesses = guesses.values

        # Getting the point estimates
        stat = ta.clf_metrics(targets,
                              guesses,
                              cutpoint=cutpoint,
                              average=average,
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
        with Pool() as p:
            boot_input = [(targets, None, None, seed) for seed in seeds]
            boots = p.starmap(ta.boot_sample, boot_input)
            inputs = [(targets[boot], guesses[boot], average, cutpoint)
                      for boot in boots]

            # Getting the bootstrapped metrics from the Pool
            p_output = p.starmap(ta.clf_metrics, inputs)
            p.close()

        scores = pd.concat(p_output, axis=0)
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
            j = jackknife_metrics(targets, guesses, cutpoint, average)
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
                np.nanpercentile(scores.iloc[:, i],
                                 q=(lower_q[i], upper_q[i]),
                                 interpolation=interpolation,
                                 axis=0) for i in range(len(lower_q))
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
    with Pool() as p:
        boot_input = [(targets, sample_by, None, seed) for seed in seeds]
        boots = p.starmap(ta.boot_sample, boot_input)

        # Getting the ROC curves
        roc_input = [(targets[boot], scores[boot]) for boot in boots]
        rocs = p.starmap(roc_curve, roc_input)

        p.close()

    return rocs
