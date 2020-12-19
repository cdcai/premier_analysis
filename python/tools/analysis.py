import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy.stats import binom, chi2, norm
from copy import deepcopy
from multiprocessing import Pool
from copy import deepcopy


# Quick function for thresholding probabilities
def threshold(probs, cutoff=.5):
    return np.array(probs >= cutoff).astype(np.uint8)


# Calculates McNemar's chi-squared statistic
def mcnemar_test(true, pred, cc=True):
    cm = confusion_matrix(true, pred)
    b = int(cm[0, 1])
    c = int(cm[1, 0])
    if cc:
        stat = (abs(b - c) - 1)**2 / (b + c)
    else:
        stat = (b - c)**2 / (b + c)
    p = 1 - chi2(df=1).cdf(stat)
    outmat = np.array([b, c, stat, p]).reshape(-1, 1)
    out = pd.DataFrame(outmat.transpose(),
                       columns=['b', 'c', 'stat', 'pval'])
    return out


# Calculates the Brier score for multiclass problems
def brier_score(true, pred):
    return np.sum((pred - true)**2) / true.shape[0]


# Slim version of clf_metrics
def slim_metrics(df, rules, by=None):
    if by is not None:
        good_idx = np.where(by == 1)[0]
        df = df.iloc[good_idx]
    N = df.shape[0]
    out = np.zeros(shape=(len(rules), 2))
    for i, rule in enumerate(rules):
        out[i, 0] = np.sum(df[rule])
        out[i, 1] = out[i, 0] / N
    out_df = pd.DataFrame(out, columns=['n', 'pct'])
    out_df['rule'] = rules
    out_df = out_df[['rule', 'n', 'pct']]
    return out_df


# Runs basic diagnostic stats on binary (only) predictions
def clf_metrics(true, 
                pred,
                average_by=None,
                weighted=True,
                round=4,
                round_pval=False,
                mcnemar=False):
    
    # Converting pd.Series to np.array
    stype = type(pd.Series())
    if type(pred) == stype:
        pred = pred.values
    if type(true) == stype:
        true = true.values
    if type(average_by) == stype:
        average_by == average_by.values
    
    # Optionally returning macro-average results
    if average_by is not None:
        return macro_clf_metrics(targets=true,
                                 guesses=pred,
                                 by=average_by,
                                 weighted=weighted,
                                 round=round)
    # Constructing the 2x2 table
    confmat = confusion_matrix(true, pred)
    tp = confmat[1, 1]
    fp = confmat[0, 1]
    tn = confmat[0, 0]
    fn = confmat[1, 0]
    
    # Calculating basic measures of diagnostic accuracy
    sens = np.round(tp / (tp + fn), round)
    spec = np.round(tn / (tn + fp), round)
    ppv = np.round(tp / (tp + fp), round)
    npv = np.round(tn / (tn + fn), round)
    f1 = np.round(2 * (sens * ppv) / (sens + ppv), round)
    j = sens + spec - 1
    mcc_num = ((tp * tn) - (fp * fn))
    mcc_denom = np.sqrt(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    mcc = mcc_num / mcc_denom
    brier = np.round(brier_score(true, pred), round)
    outmat = np.array([tp, fp, tn, fn,
                       sens, spec, ppv,
                       npv, j, f1, mcc, brier]).reshape(-1, 1)
    out = pd.DataFrame(outmat.transpose(),
                       columns=['tp', 'fp', 'tn', 
                                'fn', 'sens', 'spec', 'ppv',
                                'npv', 'j', 'f1', 'mcc', 'brier'])
    
    # Calculating some additional measures based on positive calls
    true_prev = int(np.sum(true == 1))
    pred_prev = int(np.sum(pred == 1))
    abs_diff = (true_prev - pred_prev) * -1
    rel_diff = np.round(abs_diff / true_prev, round)
    if mcnemar:
        pval = mcnemar_test(true, pred).pval[0]
        if round_pval:
            pval = np.round(pval, round)
    count_outmat = np.array([true_prev, pred_prev, abs_diff, 
                             rel_diff]).reshape(-1, 1)
    count_out = pd.DataFrame(count_outmat.transpose(),
                             columns=['true_prev', 'pred_prev', 
                                      'prev_diff', 'rel_prev_diff'])
    out = pd.concat([out, count_out], axis=1)
    
    # Optionally dropping the mcnemar p-val
    if mcnemar:
        out['mcnemar'] = pval
    
    return out


# Performs either macro or weighted macro averaging of clf_metrics
def macro_clf_metrics(targets,
                      guesses,
                      by,
                      weighted=True,
                      round=4,
                      p_method='harmonic',
                      mcnemar=True):
    # Column groups for rounding later
    count_cols = ['tp', 'fp', 'tn', 'fn']
    prev_cols = ['true_prev', 'pred_prev', 'prev_diff']
    
    # Getting the indices for each group
    n = len(targets)
    group_names = np.unique(by)
    n_groups = len(group_names)
    group_idx = [np.where(by == group)[0]
                 for group in group_names]
    group_counts = np.array([len(idx) for idx in group_idx])
    
    # Calculating the groupwise statistics
    group_stats = [clf_metrics(targets[idx],
                               guesses[idx],
                               mcnemar=mcnemar) 
                   for idx in group_idx]
    
    # Casting the basic counts as proportions
    for i, df in enumerate(group_stats):
        df[count_cols] /= group_counts[i]
        df[prev_cols] /= group_counts[i]
    
    group_stats = pd.concat(group_stats, axis=0)
    
    # Calculating the weights
    if weighted:
        w = np.array(group_counts / n)
    else:
        w = np.repeat(1 / n_groups, n_groups)
    
    # Calculating the mean values
    averages = np.average(group_stats, axis=0, weights=w)
    avg_stats = pd.DataFrame(averages).transpose()
    avg_stats.columns = group_stats.columns.values
    
    # Converting the count metrics back to integers
    avg_stats[count_cols] *= n
    avg_stats[count_cols] = avg_stats[count_cols].astype(int)
    avg_stats[prev_cols] *= n
    avg_stats.rel_prev_diff = avg_stats.prev_diff / avg_stats.true_prev
    
    # Rounding off the floats
    float_cols = ['sens', 'spec', 'npv', 
                  'ppv', 'j', 'f1', 'brier']
    avg_stats[float_cols] = avg_stats[float_cols].round(round)
    avg_stats.rel_prev_diff = avg_stats.rel_prev_diff.round(round)
    
    # Getting the mean of the p-values with either Fisher's method
    # or the harmonic mean method
    if mcnemar:
        avg_stats.mcnemar = average_pvals(group_stats.mcnemar,
                                          w=w,
                                          method=p_method)
    
    return avg_stats


def average_pvals(p_vals, 
                  w=None, 
                  method='harmonic',
                  smooth=True,
                  smooth_val=1e-7):
    if smooth:
        p = p_vals + smooth_val
    else:
        p = deepcopy(p_vals)
    if method == 'harmonic':
        if w is None:
            w = np.repeat(1 / len(p), len(p))
        p_avg = 1 / np.sum(w / p)
    elif method == 'fisher':
        stat = -2 * np.sum(np.log(p))
        p_avg = 1 - chi2(df=1).cdf(stat)
    return p_avg


# Generates bootstrap indices of a dataset with the option
# to stratify by one of the (binary-valued) variables
def boot_sample(df,
                by=None,
                size=None,
                seed=None,
                return_df=False):
    
    # Setting the random states for the samples
    if seed is None:
        seed = np.random.randint(1, 1e6, 1)[0]
    np.random.seed(seed)
    
    # Getting the sample size
    if size is None:
        size = df.shape[0]
    
    # Sampling across groups, if group is unspecified
    if by is None:
        np.random.seed(seed)
        idx = range(size)
        boot = np.random.choice(idx,
                                size=size,
                                replace=True)
    
    # Sampling by group, if group has been specified
    else:
        levels = np.unique(by)
        level_idx = [np.where(by == level)[0]
                     for level in levels]
        boot = np.random.choice(level_idx,
                                size=len(levels),
                                replace=True) 
        boot = np.concatenate(boot).ravel()
    
    if not return_df:
        return boot
    else:
        return df.iloc[boot, :]
    

def diff_boot_cis(ref, 
                  comp, 
                  a=0.05,
                  abs_diff=False, 
                  method='bca',
                  interpolation='nearest'):
    # Quick check for a valid estimation method
    methods = ['pct', 'diff', 'bca']
    assert method in methods, 'Method must be pct, diff, or bca.'
    
    # Pulling out the original estiamtes
    ref_stat = pd.Series(ref.cis.stat.drop('true_prev').values)
    ref_scores = ref.scores.drop('true_prev', axis=1)
    comp_stat = pd.Series(comp.cis.stat.drop('true_prev').values)
    comp_scores = comp.scores.drop('true_prev', axis=1)
    
    # Optionally Reversing the order of comparison
    diff_scores = comp_scores - ref_scores
    diff_stat = comp_stat - ref_stat
        
    # Setting the quantiles to retrieve
    lower = (a / 2) * 100
    upper = 100 - lower
    
    # Calculating the percentiles 
    if method == 'pct':
        cis = np.nanpercentile(diff_scores,
                               q=(lower, upper),
                               interpolation=interpolation,
                               axis=0)
        cis = pd.DataFrame(cis.transpose())
    
    elif method == 'diff':
        diffs = diff_stat.values.reshape(1, -1) - diff_scores
        percents = np.nanpercentile(diffs,
                                    q=(lower, upper),
                                    interpolation=interpolation,
                                    axis=0)
        lower_bound = pd.Series(diff_stat + percents[0])
        upper_bound = pd.Series(diff_stat + percents[1])
        cis = pd.concat([lower_bound, upper_bound], axis=1)
    
    elif method == 'bca':
        # Removing true prevalence from consideration to avoid NaNs
        ref_j_means = ref.jack[1].drop('true_prev')
        ref_j_scores = ref.jack[0].drop('true_prev', axis=1)
        comp_j_means = comp.jack[1].drop('true_prev')
        comp_j_scores = comp.jack[0].drop('true_prev', axis=1)
        
        # Calculating the bias-correction factor
        n = ref.scores.shape[0]
        stat_vals = diff_stat.transpose().values.ravel()
        n_less = np.sum(diff_scores < stat_vals, axis=0)
        p_less = n_less / n
        z0 = norm.ppf(p_less)
        
        # Fixing infs in z0
        z0[np.where(np.isinf(z0))[0]] = 0.0
        
        # Estiamating the acceleration factor
        j_means = comp_j_means - ref_j_means
        j_scores = comp_j_scores - ref_j_scores
        diffs = j_means - j_scores
        numer = np.sum(np.power(diffs, 3))
        denom = 6 * np.power(np.sum(np.power(diffs, 2)), 3/2)
        
        # Getting rid of 0s in the denominator
        zeros = np.where(denom == 0)[0]
        for z in zeros:
            denom[z] += 1e-6
        
        acc = numer / denom
        
        # Calculating the bounds for the confidence intervals
        zl = norm.ppf(a / 2)
        zu = norm.ppf(1 - (a/2))
        lterm = (z0 + zl) / (1 - acc*(z0 + zl))
        uterm = (z0 + zu) / (1 - acc*(z0 + zu))
        lower_q = norm.cdf(z0 + lterm) * 100
        upper_q = norm.cdf(z0 + uterm) * 100
                                
        # Returning the CIs based on the adjusted quantiles
        cis = [np.nanpercentile(diff_scores.iloc[:, i], 
                                q=(lower_q[i], upper_q[i]),
                                interpolation=interpolation,
                                axis=0) 
               for i in range(len(lower_q))]
        cis = pd.DataFrame(cis, columns=['lower', 'upper'])
                
    cis = pd.concat([ref_stat, comp_stat, diff_stat, cis], 
                    axis=1)
    cis = cis.set_index(ref_scores.columns.values)
    cis.columns = ['ref', 'comp', 'd', 
                   'lower', 'upper']
    
    return cis


def grid_metrics(targets,
                 guesses,
                 step=.01,
                 min=0.0,
                 max=1.0,
                 by='f1',
                 average='binary',
                 counts=True):
    cutoffs = np.arange(min, max, step)
    if len((guesses.shape)) == 2:
        if guesses.shape[1] == 1:
            guesses = guesses.flatten()
        else:
            guesses = guesses[:, 1]
    if average == 'binary':
        scores = []
        for i, cutoff in enumerate(cutoffs):
            threshed = threshold(guesses, cutoff)
            stats = clf_metrics(targets, threshed)
            stats['cutoff'] = pd.Series(cutoff)
            scores.append(stats)
    
    return pd.concat(scores, axis=0)


def roc_cis(rocs, alpha=0.05, round=2):
    # Getting the quantiles to make CIs
    lq = (alpha / 2) * 100
    uq = (1 - (alpha / 2)) * 100
    fprs = np.concatenate([roc[0] for roc in rocs], axis=0)
    tprs = np.concatenate([roc[1] for roc in rocs], axis=0)
    roc_arr = np.concatenate([fprs.reshape(-1, 1), 
                              tprs.reshape(-1, 1)], 
                             axis=1)
    roc_df = pd.DataFrame(roc_arr, columns=['fpr', 'tpr'])
    roc_df.fpr = roc_df.fpr.round(round)
    unique_fprs = roc_df.fpr.unique()
    fpr_idx = [np.where(roc_df.fpr == fpr)[0] for fpr in unique_fprs]
    tpr_quants = [np.percentile(roc_df.tpr[idx], q=(lq, 50, uq)) 
                  for idx in fpr_idx]
    tpr_quants = np.vstack(tpr_quants)
    quant_arr = np.concatenate([unique_fprs.reshape(-1, 1),
                                tpr_quants],
                               axis=1)
    quant_df = pd.DataFrame(quant_arr, columns=['fpr', 'lower',
                                                'med', 'upper'])
    quant_df = quant_df.sort_values('fpr')
    return quant_df


# Returns the maximum value of metric X that achieves a value of
# at least yval on metric Y
def x_at_y(x, y, yval, grid):
    y = np.array(grid[y])
    x = np.array(grid[x])
    assert np.sum(y >= yval) > 0, 'No y vals meet the minimum'
    good_y = np.where(y >= yval)[0]
    best_x = np.max(x[good_y])
    return best_x


# Converts a boot_cis['cis'] object to a single row
def merge_cis(df, stats, round=4):
    df = deepcopy(df)
    for stat in stats:
        lower = stat + '.lower'
        upper = stat + '.upper'
        new = stat + '.ci'
        l = df[lower].values.round(round)
        u = df[upper].values.round(round)
        strs = [pd.Series('(' + str(l[i]) + ', ' + str(u[i]) + ')')
                for i in range(df.shape[0])]
        df[new] = pd.concat(strs, axis=0)
        df = df.drop([lower, upper], axis=1)
    return df


def unique_combo(c):
    if len(np.intersect1d(c[0], c[1])) == 0:
        return c
    else:
        return None


def prop_table(y, pred, axis=0, round=2):
    tab = pd.crosstab(y, pred)
    if axis == 1:
        tab = tab.transpose()
        out = tab / np.sum(tab, axis=0)
        out = out.transpose()
    else:
        out = tab / np.sum(tab, axis=0)
    if round is not None:
        out = np.round(out, round)
    return out
        

def risk_ratio(y, pred, round=2):
    props = np.array(prop_table(y, pred, round=None))
    rr = props[1, 1] / props[1, 0]
    if round is not None:
        rr = np.round(rr, round)
    return rr


def odds_ratio(y, pred, round=2):
    tab = np.array(pd.crosstab(y, pred))
    OR = (tab[0, 0]*tab[1, 1]) / (tab[1, 0]*tab[0, 1])
    if round is not None:
        OR = np.round(OR, round)
    return OR

