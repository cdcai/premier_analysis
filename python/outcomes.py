import numpy as np
import pandas as pd
import os


# Setting the file directories
prem_dir = 'data/data/'
out_dir = 'output/'

# Importing the parquet files
icd = pd.read_parquet(prem_dir + 'icdcode/')
providers = pd.read_parquet(prem_dir + 'providers/')
pat = pd.read_parquet(prem_dir + 'vw_covid_pat/')
pat_diag = pd.read_parquet(prem_dir + 'vw_covid_paticd_diag/')
pat_proc = pd.read_parquet(prem_dir + 'vw_covid_paticd_proc/')
genlab = pd.read_parquet(prem_dir + 'vw_covid_genlab/')
lab_res = pd.read_parquet(prem_dir + 'vw_covid_lab_res/')
vitals = pd.read_parquet(prem_dir + 'vw_covid_vitals/')
bill_pharm = pd.read_parquet(prem_dir + 'vw_covid_bill_pharm/')
bill_oth = pd.read_parquet(prem_dir + 'vw_covid_bill_oth/')

# List of unique visit ids
outcomes = pd.DataFrame(np.unique(pat.pat_key), columns=['pat_key'])

# Adding mechanical ventilation
vented = np.where(['VENTILA' in doc for doc in bill_oth.std_chg_desc])[0]
vent_ids = bill_oth.pat_key[vented]
outcomes['vented'] = np.array([id in vent_ids.values
                               for id in outcomes.pat_key],
                            dtype=np.uint8)

# Adding death as a target to the patient table
died = pat[pat.disc_status_desc == 'EXPIRED'].reset_index()
outcomes['died'] = np.array([id in died.pat_key.values 
                             for id in outcomes.pat_key],
                            dtype=np.uint8)

# Making some targets for different lengths of stay
los_7 = pat[pat.los > 7]
los_14 = pat[pat.los > 14]
los_28 = pat[pat.los > 28]

outcomes['los_7'] = np.array([id in los_7.pat_key.values
                              for id in outcomes.pat_key], 
                             dtype=np.uint8)
outcomes['los_14'] = np.array([id in los_14.pat_key.values
                               for id in outcomes.pat_key], 
                              dtype=np.uint8)
outcomes['los_28'] = np.array([id in los_28.pat_key.values
                               for id in outcomes.pat_key], 
                              dtype=np.uint8)

# Making columns for SARS test results
sars_pcr = np.array([
    'SARS' in lab_res.test[i] and 
    'RNA' in lab_res.test[i] and
    'posit' in lab_res.observation[i]
    for i in range(lab_res.shape[0])],
                    dtype=np.uint8)
sars_ser = np.array([
    'SARS' in lab_res.test[i] and
    'Ab.' in lab_res.test[i] and
    'posit' in lab_res.observation[i]
    for i in range(lab_res.shape[0])],
                    dtype=np.uint8)
sars_labs = pd.DataFrame([lab_res.pat_key,
                          pd.Series(sars_pcr),
                          pd.Series(sars_ser)]).transpose()
sars_labs.columns = ['pat_key', 'sars_pcr', 'sars_ser']
pcr_pos = sars_labs.pat_key[sars_labs.sars_pcr == 1]
ser_pos = sars_labs.pat_key[sars_labs.sars_ser == 1]

# Merging the outcomes back into the pat table
outcomes['sars_pcr'] = np.array([id in pcr_pos.values 
                                 for id in outcomes.pat_key],
                                dtype=np.uint8)
outcomes['sars_ser'] = np.array([id in ser_pos.values 
                                 for id in outcomes.pat_key],
                                dtype=np.uint8)

# Adding diagnosis info to the other outcomes
icu = np.where(['ICU' in doc for doc in bill_oth.std_chg_desc])[0]
icu_ids = icu_df.pat_key[icu]
outcomes['icu'] = np.array([id in icu_ids.values
                            for id in outcomes.pat_key],
                           dtype=np.uint8)

# Adding the covid case definition to the other outcomes
covid = np.array(['U07.1' in doc or 'B97.29' in doc
                  for doc in pat_diag.icd_code],
                 dtype=np.uint8)
covid_df = pd.concat([pat_diag.pat_key, pd.Series(covid)], axis=1)
covid_df.columns = ['pat_key', 'covid']
covid_ids = covid_df.pat_key[covid_df.covid == 1]
outcomes['covid'] = np.array([id in covid_ids.values
                            for id in outcomes.pat_key],
                           dtype=np.uint8)

# Writing the outcomes alone
outcomes.to_csv(out_dir + 'outcomes.csv', index=False)
