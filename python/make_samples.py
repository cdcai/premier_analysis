import numpy as np
import pandas as pd
import os

# Setting the file directories
prem_dir = 'data/data/'
out_dir = 'output/samples/'
parq_dir = out_dir + 'parquet/'
pkl_dir = out_dir + 'pkl/'

# Read in a parquet file, sample the first N rows, then write to csv
files = [f for f in os.listdir(prem_dir) if 'parquet' in f]
views = [f for f in files if 'vw' in f]

for view in views:
    p = pd.read_parquet(prem_dir + view)
    if 'medrec_key' in p.columns.values:
        p.sort_values(['medrec_key', 'pat_key'], inplace=True)
    else:
        p.sort_values('pat_key', inplace=True)
    csv_name = view.replace('parquet', 'csv')
    p.head(1000).to_csv(out_dir + csv_name, index=False)

