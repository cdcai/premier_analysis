import argparse
import os

import numpy as np
import pandas as pd

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

    # Writing the confidence intervals to disk
    with pd.ExcelWriter(stats_dir + OUTCOME + '_cis.xlsx') as writer:
        for i, obj in enumerate(cis):
            obj.cis.to_excel(writer, sheet_name=mods[i])
        writer.save()
