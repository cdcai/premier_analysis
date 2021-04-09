'''This script generates bootstrap CIs for the models and metrics'''

import numpy as np
import pandas as pd
import pickle
import os

import tools.analysis as ta
import tools.multi as tm


# Setting the directories
output_dir = os.path.abspath('output/') + '/'
stats_dir = output_dir + 'analysis/'

# Importing the predictions
death_preds = pd.read_csv(stats_dir + 'death_preds.csv')
multi_preds = pd.read_csv(stats_dir + 'multi_class_preds.csv')
misa_pt_preds = pd.read_csv(stats_dir + 'misa_pt_preds.csv')

# Setting the models to look at
mods = ['lgr_d1', 'rf_d1', 'gbc_d1', 'svm_d1', 'dan_d1', 'lstm']
pred_dfs = [death_preds, multi_preds, misa_pt_preds]
outcomes = ['death', 'multi_class', 'misa_pt']
cis = []

# Running the single confidence intervals (this takes a little time)
probs_dir = stats_dir + 'probs/'
prob_files = os.listdir(probs_dir)
for i, outcome in enumerate(outcomes):
    outcome_cis = []
    for mod in mods:
        mod_probs = mod + '_' + outcome + '.pkl'
        if mod_probs in prob_files:
            with open(probs_dir + mod_probs, 'rb') as f:
                prob_dict = pickle.load(f)
                cutpoint = prob_dict['cutpoint']
                guesses = prob_dict['probs']    
        else:
            cutpoint = 0.5
            guesses = pred_dfs[i][mod + '_pred']
        
        ci = ta.boot_cis(targets=pred_dfs[i][outcome],
                         guesses=guesses,
                         cutpoint=cutpoint,
                         n=100)
        outcome_cis.append(ci)
    cis.append(ta.merge_ci_list(outcome_cis, 
                                mod_names=mods,
                                round=2))

# Writing the confidence intervals to disk
writer = pd.ExcelWriter(stats_dir +'cis.xlsx')
for i, outcome in enumerate(outcomes):
    cis[i].to_excel(writer, sheet_name=outcome)

writer.save()
