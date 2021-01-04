'''This script takes the full list of lists of visits and prepares them for
modeling, e.g., by cutting them to specific lengths and specifying their
labels.
'''

import numpy as np
import pandas as pd
import pickle as pkl
import os

from importlib import reload
from multiprocessing import Pool

import tools.preprocessing as tp

# Which COVID visit to use as the focus for prediction--the first, the last,
# or both.
CUT_METHOD = 'first'

# Time in days to the prediction horizon from the start of the final visit
HORIZON = 1

# Maximum length of lookback period
MAX_SEQ = 225

# Pat-level outcome to use as the label
OUTCOME = 'misa_pt'

# %% Setting the directories
output_dir = os.path.abspath('output/') + '/'
data_dir = os.path.abspath('..data/data/') + '/'
pkl_dir = output_dir + 'pkl/'


# %%
def main():

    # Reading in the full dataset
    with open(pkl_dir + 'int_seqs.pkl', 'rb') as f:
        int_seqs = pkl.load(f)

    with open(pkl_dir + 'pat_data.pkl', 'rb') as f:
        pat_data = pkl.load(f)

    # Total number of patients
    n_patients = len(int_seqs)

    # Trimming the day sequences
    with Pool() as p:
        # Finding the cut points for the sequences
        find_input = [(pat_data['covid'][i], pat_data['length'][i], HORIZON,
                       MAX_SEQ, CUT_METHOD) for i in range(n_patients)]
        cut_points = p.starmap(tp.find_cutpoints, find_input)

        # Figuring out who doesn't have another day after the horizon
        keepers = [
            pat_data['length'][i][cut_points[i][1]] > 1
            and pat_data['inpat'][i][cut_points[i][1]] == 1
            and pat_data['age'][i][cut_points[i][1]] > 17
            for i in range(n_patients)
        ]

        # Trimming the inputs and outputs to the right length
        trim_input = [(int_seqs[i], pat_data[OUTCOME][i], cut_points[i])
                      for i in range(n_patients)]
        trim_out = p.starmap(tp.trim_sequence, trim_input)

        # Keeping the keepers and booting the rest
        trim_out = [trim_out[i] for i in range(n_patients) if keepers[i]]

    # output max time to use in keras model
    print("Use TIME_SEQ:{}".format(max([len(x) for x, _, _ in trim_out])))

    # Saving the trimmed sequences to disk
    with open(pkl_dir + 'trimmed_seqs.pkl', 'wb') as f:
        pkl.dump(trim_out, f)


# %%
if __name__ == "__main__":
    main()