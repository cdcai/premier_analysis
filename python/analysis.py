import argparse
import os

import numpy as np
import pandas as pd
import pickle
import os

import tools.analysis as ta
import tools.multi as tm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--day_one',
        help="Use only first inpatient day's worth of features (DAN only)",
        dest='day_one',
        action='store_true')
    parser.add_argument('--all_days',
                        help="Use all features in lookback period (DAN only)",
                        dest='day_one',
                        action='store_false')
    parser.set_defaults(day_one=True)
    parser.add_argument("--outcome",
                        type=str,
                        default="misa_pt",
                        choices=["misa_pt", "multi_class", "death"],
                        help="which outcome to use as the prediction target")

    args = parser.parse_args()

    DAY_ONE = args.day_one
    OUTCOME = args.outcome

    # Setting the directories
    pwd = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(pwd, "..", "output", "")
    stats_dir = os.path.join(output_dir, "analysis", "")

    # Path where the metrics will be written
    ci_file = os.path.join(stats_dir, OUTCOME + "_cis.xlsx")

    # Importing the predictions
    preds = pd.read_csv(stats_dir + OUTCOME + "_preds.csv")

    # Setting the models to look at
    mods = ['lgr', 'rf', 'gbc', 'svm', 'dan', 'lstm']

    if DAY_ONE:
        mods = [mod + "_d1" for mod in mods if mod != "lstm"]

    # Running the single confidence intervals (this takes a little time)
    cis = [
        tm.boot_cis(preds[OUTCOME], preds[mod + '_pred'], n=1000)
        for mod in mods
    ]

# Setting the directories
output_dir = os.path.abspath('output/') + '/'
stats_dir = output_dir + 'analysis/'

# Importing the predictions
death_preds = pd.read_csv(stats_dir + 'death_preds.csv')
multi_preds = pd.read_csv(stats_dir + 'multi_class_preds.csv')

# Setting the models to look at
mods = ['lgr_d1', 'rf_d1', 'gbc_d1', 'svm_d1', 'dan_d1']
pred_dfs = [death_preds, multi_preds]
outcomes = ['death', 'multi', 'misa_pt']
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
                guesses = pickle.load(f)
        else:
            guesses = pred_dfs[i][mod + '_pred']
            
        ci = ta.boot_cis(targets=pred_dfs[i][outcome],
                         guesses=guesses,
                         n=100)
        pred_cis.append(ci)
    cis.append(ta.merge_ci_list(outcome_cis, mod_names=mods))

# Writing the confidence intervals to disk
writer = pd.ExcelWriter(stats_dir +'cis.xlsx')
for i, outcome in enumerate(outcomes):
    cis[i].to_excel(writer, sheet_name=outcome)

writer.save()
