# Databricks notebook source
from sklearn.metrics import confusion_matrix

def calcualte_metrics(true, pred):
    confmat = confusion_matrix(true, pred)
    tp = confmat[1, 1]
    fp = confmat[0, 1]
    tn = confmat[0, 0]
    fn = confmat[1, 0]

    # Calculating the main binary metrics
    ppv = np.round(tp / (tp + fp), round) if tp + fp > 0 else 0
    sens = np.round(tp / (tp + fn), round) if tp + fn > 0 else 0
    spec = np.round(tn / (tn + fp), round) if tn + fp > 0 else 0
    npv = np.round(tn / (tn + fn), round) if tn + fn > 0 else 0
    f1 = np.round(2 * (sens * ppv) /
                  (sens + ppv), round) if sens + ppv != 0 else 0

    # Calculating the Matthews correlation coefficient
    mcc_num = ((tp * tn) - (fp * fn))
    mcc_denom = np.sqrt(((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = mcc_num / mcc_denom if mcc_denom != 0 else 0

    # Calculating Youden's J and the Brier score
    j = sens + spec - 1
    brier = np.round(brier_score(true, pred), round)

    # Rolling everything so far into a dataframe
    outmat = np.array(
        [tp, fp, tn, fn, sens, spec, ppv, npv, j, f1, mcc,
         brier]).reshape(-1, 1)
    out = pd.DataFrame(outmat.transpose(),
                       columns=[
                           'tp', 'fp', 'tn', 'fn', 'sens', 'spec', 'ppv',
                           'npv', 'j', 'f1', 'mcc', 'brier'
                       ])
    return out
