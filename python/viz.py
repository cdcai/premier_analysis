'''Makes some basic visuals for the MIS-A analysis'''

import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve


# Import the model predictions on the test data
file_dir = 'C:/Users/yle4/code/github/premier_analysis/python/output/analysis/'
out_dir = file_dir + 'figures/'
outcomes = ['icu', 'misa_pt', 'death']
icu_preds = pd.read_csv(file_dir + 'icu_preds.csv')
misa_preds = pd.read_csv(file_dir + 'misa_pt_preds.csv')
death_preds = pd.read_csv(file_dir + 'death_preds.csv')
prob_cols = ['lgr_d1_prob', 'rf_d1_prob', 'gbc_d1_prob',
             'dan_d1_prob', 'lstm_prob']
mod_names = ['lgr', 'rf', 'gbc', 'dan', 'lstm']
fig_titles = ['ICU', 'MIS-A', 'death']

# Running the ROC curves
sns.set_palette('colorblind')
sns.set_style('dark')
for i, df in enumerate(pred_dfs):
    sk_rocs = [roc_curve(df[outcomes[i]],
                         df[col]) for col in prob_cols]
    roc_dfs = [pd.DataFrame([r[0], r[1]]).transpose()
               for r in sk_rocs]
    for j, roc_df in enumerate(roc_dfs):
        roc_df['model'] = mod_names[j]
    all_dfs = pd.concat(roc_dfs, axis=0)
    all_dfs.columns = ['fpr', 'tpr', 'model']
    sns.lineplot(x='fpr', y='tpr', data=all_dfs, hue='model', ci=None)
    sns.lineplot(x=(0, 1), y=(0, 1), color='lightgray')
    plt.title(fig_titles[i])
    plt.savefig(out_dir + outcomes[i] + '_' + 'ROC.pdf',
                bbox_inches='tight')
    plt.clf()

# Running the histograms
for i, df in enumerate(pred_dfs):
    pred_probs = [pd.DataFrame(df[[col]]) for col in prob_cols]
    for j, p_df in enumerate(pred_probs):
        p_df['model'] = mod_names[j]
        p_df.columns = ['pred_prob', 'model']
    all_dfs = pd.concat(pred_probs, axis=0)
    plt.title(fig_titles[i])
    sns.histplot(x=all_dfs.pred_prob, hue=all_dfs.model)
    plt.savefig(out_dir + outcomes[i] + '_' + 'hist.pdf',
                bbox_inches='tight')
    plt.clf()
