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

# Setting the directories
output_dir = os.path.abspath('../output/') + '/'
data_dir = os.path.abspath('../data/data/') + '/'
pkl_dir = output_dir + 'pkl/'

# Reading in the full dataset
int_seqs = pkl.load(open(pkl_dir + 'int_seqs.pkl', 'rb'))
pat_data = pkl.load(open(pkl_dir + 'pat_data.pkl', 'rb'))

# Total number of patients
n_patients = len(int_seqs)

# Finding the cut points for the day sequences
p = Pool()
find_input = [(pat_data['cv_pats'][i],
               pat_data['pat_lengths'][i],
               HORIZON,
               CUT_METHOD)
           for i in range(n_patients)]
cut_points = p.starmap(tp.find_cutpoints, find_input)

# Trimming the inputs and outputs to the right length
trim_input = [(int_seqs[i], 
               pat_data['pat_deaths'][i],
               cut_points[i])
              for i in range(n_patients)]
trim_out = p.starmap(tp.trim_sequence, trim_input)

