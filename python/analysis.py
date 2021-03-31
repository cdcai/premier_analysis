import numpy as np
import pandas as pd
import os

import tools.analysis as ta
import tools.multi as tm


# Setting the directories
output_dir = os.path.abspath('output/') + '/'
stats_dir = output_dir + 'analysis/'

# Importing the predictions
death_preds = pd.read_csv(stats_dir + 'death_preds.csv')
multi_preds = pd.read_csv(stats_dir + 'multi_class_preds.csv')

# Setting the models to look at
mods = ['lgr_d1', 'rf_d1', 'gbc_d1', 'svm_d1', 'dan_d1']

# Running the single confidence intervals (this takes a little time)
death_cis = [ta.boot_cis(death_preds.death,
                         death_preds[mod + '_pred'],
                         n=1000) 
             for mod in mods]
multi_cis = [ta.boot_cis(multi_preds.multi_class,
                         multi_preds[mod + '_pred'],
                         n=1000) 
             for mod in mods]
cis = [ta.merge_ci_list(death_cis, mod_names=mods), 
       ta.merge_ci_list(multi_cis, mod_names=mods)]
outcomes = ['death', 'multi']

# Writing the confidence intervals to disk
writer = pd.ExcelWriter(stats_dir +'cis.xlsx')
for i, outcome in enumerate(outcomes):
    cis[i].to_excel(writer, sheet_name=outcome)

writer.save()
