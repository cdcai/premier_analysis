'''Makes some basic visuals for the MIS-A analysis'''

import pandas as pd
import numpy as np
import os
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

# Import raw data for feature plots
output_dir = os.path.abspath("output/") + "/"
data_dir = os.path.abspath("..data/data/") + "/"
tensorboard_dir = os.path.abspath("../data/model_checkpoints/") + "/"
pkl_dir = output_dir + "pkl/"
stats_dir = output_dir + 'analysis/'

with open(pkl_dir + "trimmed_seqs.pkl", "rb") as f:
    inputs = pkl.load(f)

with open(pkl_dir + "all_ftrs_dict.pkl", "rb") as f:
    vocab = pkl.load(f)

with open(pkl_dir + "feature_lookup.pkl", "rb") as f:
    all_feats = pkl.load(f)

with open(pkl_dir + "demog_dict.pkl", "rb") as f:
    demog_dict = pkl.load(f)
    demog_dict = {k:v for v, k in demog_dict.items()}

# Separating the inputs and labels
features = [t[0] for t in inputs]
demog = [t[1] for t in inputs]
cohort = pd.read_csv(output_dir + 'cohort.csv')

# Counts to use for loops and stuff
n_patients = len(features)
n_features = np.max(list(vocab.keys()))
n_classes = len(np.unique(labels))

pred_dfs = [icu_preds, misa_preds, death_preds]
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
    plt.savefig(out_dir + outcomes[i] + '_' + 'roc.png',
                bbox_inches='tight')
    plt.clf()

# Running the histograms
for i, df in enumerate(pred_dfs):
    fig, ax = plt.subplots(ncols=5, 
                           sharey=True, 
                           figsize=(15, 5),
                           tight_layout=True)
    fig.suptitle(outcomes[i])
    pred_probs = [pd.DataFrame(df[[outcomes[i], col]]) for col in prob_cols]
    for j, p_df in enumerate(pred_probs):
        #plt.title(mod_names[j])
        sns.histplot(ax=ax[j], x=p_df.iloc[:, 1], hue=p_df.iloc[:, 0])
        ax[j].set_title(mod_names[j])
    file_name = outcomes[i]
    plt.savefig(out_dir + file_name + '_hist.png', bbox_inches='tight')
    plt.clf()

# Making the descriptive plots
flat_features = [tp.flatten(l) for l in features]
flat_sets = [set(l) for l in flat_features]
fts_per_pt = np.array([len(l) for l in flat_sets])

fts_24 = [set(l[-1]) for l in features]
fts_per_pt_24 = np.array([len(l) for l in fts_24])
ft_diffs = np.array(fts_per_pt - fts_per_pt_24)

tab_names = ['vtl', 'bill', 'lbrs', 'genl', 'proc']
tab_vocab = np.array([np.sum([s in v for v in list(vocab.values())])
                      for s in tab_names])
tab_vocab = pd.DataFrame(tab_vocab.reshape(-1, 1).transpose(), 
                         columns=tab_names, 
                         dtype=np.uint16)
fts_by_tab = np.array([[np.sum([s in vocab[k] for k in l])
                               for l in flat_sets] 
                       for s in tab_names])
fts_by_tab = pd.DataFrame(fts_by_tab.transpose(),
                          columns=tab_names,
                          dtype=np.uint16)
fts_by_tab.to_csv(output_dir + 'fts_by_tab.csv', index=False)

fts_by_tab_24 = np.array([[np.sum([s in vocab[k] for k in l])
                                  for l in fts_24]
                          for s in tab_names])
fts_by_tab_24 = pd.DataFrame(fts_by_tab_24.transpose(),
                             columns=tab_names,
                             dtype=np.uint16)
fts_by_tab_24.to_csv(output_dir + 'fts_by_tab_24.csv', index=False)
