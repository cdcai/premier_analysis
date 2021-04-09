'''This script generates bootstrap CIs for the models and metrics'''
import argparse
import os

import pandas as pd
import pickle
import os

import tools.analysis as ta
import tools.multi as tm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--day_one',
        help=
        "Also include models that only evaluated first inpatient day's worth of features",
        dest='day_one',
        action='store_true')
    parser.add_argument(
        '--all_days',
        help=
        "Only compute CIs for models that used all features in lookback period",
        dest='day_one',
        action='store_false')
    parser.set_defaults(day_one=True)
    parser.add_argument("--outcome",
                        type=str,
                        default=["misa_pt", "multi_class", "death"],
                        nargs="+",
                        choices=["misa_pt", "multi_class", "death"],
                        help="which outcome to compute CIs for (default: all)")

    args = parser.parse_args()

    DAY_ONE = args.day_one
    OUTCOME = args.outcome

    # Setting the directories
    pwd = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(pwd, "..", "output", "")
    stats_dir = os.path.join(output_dir, "analysis", "")
    probs_dir = os.path.join(stats_dir, "probs", "")

    # Path where the metrics will be written
    ci_file = os.path.join(stats_dir, "cis.xlsx")

    # Checking prob files
    prob_files = os.listdir(probs_dir)

    cis = []

    for i, outcome in enumerate(OUTCOME):
        outcome_cis = []

        # Importing the predictions
        preds = pd.read_csv(stats_dir + outcome + '_preds.csv')

        # Setting the models to look at
        mods = ['lgr', 'rf', 'gbc', 'svm', 'dan', 'lstm']

        if DAY_ONE:
            mods += [mod + "_d1" for mod in mods if mod != "lstm"]

        for mod in mods:
            mod_prob_file = mod + '_' + outcome + '.pkl'

            if mod_prob_file in prob_files:
                # In the multi_class case, we've been writing pkls
                with open(probs_dir + mod_prob_file, 'rb') as f:
                    probs_dict = pickle.load(f)

                    cutpoint = probs_dict['cutpoint']
                    guesses = probs_dict['probs']
            else:
                # Otherwise the probs will be in the excel file
                cutpoint = 0.5
                guesses = pred_dfs[i][mod + '_pred']

            # Compute CIs model-by-model
            ci = ta.boot_cis(targets=preds[outcome],
                             guesses=guesses,
                             cutpoint=cutpoint,
                             n=100)
            # Append to outcome CI list
            outcome_cis.append(ci)

        # Append all outcome CIs to master list
        cis.append(ta.merge_ci_list(outcome_cis, mod_names=mods, round=2))

    # Writing all the confidence intervals to disk
    with pd.ExcelWriter(
            ci_file, mode="a" if os.path.exists(ci_file) else "w") as writer:
        for i, outcome in enumerate(OUTCOME):
            cis[i].to_excel(writer, sheet_name=outcome)
        writer.save()
