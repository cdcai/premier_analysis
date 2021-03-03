'''This script takes the full list of lists of visits and prepares them for
modeling, e.g., by cutting them to specific lengths and specifying their
labels.
'''

import numpy as np
import pandas as pd
import pickle as pkl
import argparse
import os

from importlib import reload
from multiprocessing import Pool


def find_cutpoints(visit_type, visit_length, tail=1, origin=0, how='first'):
    '''Figures out where to cut each patient's sequence of visits.
    
    See tools.preprocessing for full docstring.
    '''
    covid_idx = np.where(np.array(visit_type) == 1)[0]
    first = np.min(covid_idx)
    first_end = np.sum(visit_length[0:first], dtype=np.uint16) + tail
    last = np.max(covid_idx)
    last_end = np.sum(visit_length[0:last], dtype=np.uint16) + tail

    if how == 'first':
        if origin != 0:
            origin = np.maximum(0, first_end - origin)
        return (origin, first_end), first
    elif how == 'last':
        if origin != 0:
            origin = np.maximum(0, last_end - origin)
        return (origin, last_end), last
    elif how == 'both':
        return (first_end, last_end), last


def trim_sequence(inputs, labels, cuts):
    '''Trims the sequences of visits according to find_cutpoints.
    
    See tools.preprocessing for full docstring.
    '''
    in_start, in_end = cuts[0][0], cuts[0][1]
    label_id = cuts[1]
    return inputs[0][in_start:in_end], inputs[1], labels[label_id]


def flatten(l):
    if type(l) != type([]):
        return l
    if type(l[0]) == type([]):
        return [item for sublist in l for item in sublist]
    else:
        return l


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cut_method',
                        type=str,
                        default='first',
                        choices=['first', 'last', 'both'],
                        help='which COVID visit(s) to use as bookends')
    parser.add_argument('--horizon',
                        type=int,
                        default=1,
                        help='prediction horizon for the target visit')
    parser.add_argument('--max_seq',
                        type=int,
                        default=225,
                        help='max number of days to include')
    parser.add_argument('--outcome',
                        type=str,
                        default='misa_pt',
                        choices=['misa_pt', 'multi_class', 'death'],
                        help='which outcome to use as the prediction target')
    parser.add_argument('--exclude_icu',
                        type=bool,
                        default=True,
                        help='whether to exclude patients in the ICU before the\
                         prediction horizon')
    parser.add_argument('--out_dir',
                        type=str,
                        default='output/',
                        help='output directory')
    parser.add_argument('--data_dir',
                        type=str,
                        default='..data/data/',
                        help='path to the Premier data')
    parser.add_argument('--min_age',
                        type=int,
                        default=18,
                        help='minimum age of patients to include')
    parser.add_argument('--max_age',
                        type=int,
                        default=120,
                        help='max age of patients to include')
    parser.add_argument('--write_df',
                        type=bool,
                        default=True,
                        help='whether to write patient data to a DF')
    args = parser.parse_args()

    # Setting the globals
    CUT_METHOD = args.cut_method
    HORIZON = args.horizon
    MAX_SEQ = args.max_seq
    OUTCOME = args.outcome
    EXCLUDE_ICU = args.exclude_icu
    MIN_AGE = args.min_age
    WRITE_DF = args.write_df

    # Setting the directories
    output_dir = os.path.abspath(args.out_dir) + '/'
    data_dir = os.path.abspath(args.data_dir) + '/'
    pkl_dir = output_dir + 'pkl/'

    # Reading in the full dataset
    with open(pkl_dir + 'int_seqs.pkl', 'rb') as f:
        int_seqs = pkl.load(f)

    with open(pkl_dir + 'pat_data.pkl', 'rb') as f:
        pat_data = pkl.load(f)
    
    with open(pkl_dir + "all_ftrs_dict.pkl", "rb") as f:
        vocab = pkl.load(f)
    
    with open(pkl_dir + "feature_lookup.pkl", "rb") as f:
        all_feats = pkl.load(f)
    
    # Total number of patients
    n_patients = len(int_seqs)

    # Trimming the day sequences
    with Pool() as p:
        # Finding the cut points for the sequences
        find_input = [(pat_data['covid'][i], pat_data['length'][i], HORIZON,
                       MAX_SEQ, CUT_METHOD) for i in range(n_patients)]
        cut_points = p.starmap(find_cutpoints, find_input)

        # Trimming the inputs and outputs to the right length
        trim_input = [(int_seqs[i], pat_data[OUTCOME][i], cut_points[i])
                      for i in range(n_patients)]
        trim_out = p.starmap(trim_sequence, trim_input)
        
        # Figuring out who has at least 1 more day after the horizon
        keepers = [
            pat_data['length'][i][cut_points[i][1]] > HORIZON
            and pat_data['inpat'][i][cut_points[i][1]] == 1
            and pat_data['age'][i][cut_points[i][1]] >= MIN_AGE
            for i in range(n_patients)
        ]
        
        # Optionally adding other exclusion criteria
        if EXCLUDE_ICU:
            rev_vocab = {v:k for k,v in vocab.items()}
            icu_codes = [k for k,v in all_feats.items() if 'ICU' in v]
            icu_ftrs = [rev_vocab[code] for code in icu_codes
                        if code in rev_vocab.keys()]
            lookback = [np.min((len(l), HORIZON)) for l in trim_out]
            first_days = [trim_out[i][0][-lookback[i]]
                          for i in range(n_patients)]
            
            # Optionally flattening the pre-horizon code lists;
            # Note: this may or may not work
            if HORIZON > 1:
                first_days = [flatten(l) for l in first_days]
            
            no_icu = [len(np.intersect1d(icu_ftrs, l)) == 0
                                   for l in first_days]
            keepers = [keepers[i] and no_icu[i] for i in range(n_patients)]
        
        # Keeping the keepers and booting the rest
        trim_out = [trim_out[i] for i in range(n_patients) if keepers[i]]
        
        # Making a DF with the pat-level data to link for analysis later
        if WRITE_DF:
            cohort = [[
                pat_data['key'][i][cut_points[i][1]],
                pat_data['age'][i][cut_points[i][1]],
                pat_data['length'][i][cut_points[i][1]],
                pat_data['misa_pt'][i][cut_points[i][1]],
                pat_data['multi_class'][i][cut_points[i][1]],
                pat_data['death'][i][cut_points[i][1]]
            ] for i in range(n_patients) if keepers[i]]
            cohort_df = pd.DataFrame(cohort)
            cohort_df.columns = [
                'key', 'age', 'length', 'misa_pt', 'multi_class', 'death'
            ]
            cohort_df.to_csv(output_dir + OUTCOME + '_cohort.csv', index=False)

    # output max time to use in keras model
    print("Use TIME_SEQ:{}".format(max([len(x) for x, _, _ in trim_out])))

    # Saving the trimmed sequences to disk
    with open(pkl_dir + OUTCOME + '_trimmed_seqs.pkl', 'wb') as f:
        pkl.dump(trim_out, f)
