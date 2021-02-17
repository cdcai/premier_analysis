# Premier Analysis  
#### Sean Browning, Scott Lee, Karen Wong  

  
Deep models using EHR data.  

## Data   

The required data is contained in a private submodule which can be recursively cloned when you pull down the repository (if you have access):  

```sh
git clone --recursive git@github.com:cdcai/premier_analysis.git
```  

Alternatively, if you've already cloned the repository and didn't pull the submodule, you can update it from the project root:  

```sh
git submodule update --init --recursive
```

## Data/Model flow  

### 1. python/feature_extraction.py  

The main pre-processing for all EHR data.

Input: Several EHR views in parquet format

Output:   

- A pandas data frame with separate columns of tokenized features by source table
    aggregated to either days, hours, or seconds from index date.  
- A feature dictionary containing the abbreviated feature names and their descriptions.

Steps:  

- Join all visit-level data across EHR views  
- Discretize continuous values by quantile (labs, vitals)
- Abbreviate categorical values to tokens (billing data, qualitative lab result)
- Aggregate all features to day, hour, or seconds from medical record index date

### 2. python/features_to_integers.py  

Additional aggregation and conversion of dataframe of string tokens by time step to integer coded nested list.


Input: Pandas dataframe from feature_extraction.py  

Output:  

  - Nested list-of-lists with all features encoded into integer representation
  - Dictionary of tokens and their encoded values  
  - dictionary of visits and their LOS and outcome indicators  

Steps:

  - Join all feature columns together into single string aggregated by timestep in visit  
  - Use sklearn CountVectorizer to encode tokens to integer representation (minimum document frequency: 5)
  - Save list-of-lists  
  - Compute LOS, outcomes for each visit, save to dictionary  

### 3. python/model_prep.py  

Trim sequences to the appropriate lookback period according to which COVID visit is of interest.

Per the working definition, this is the first visit to occur. Also labels the sequences according to the outcome of interest.

Input: 
  - List-of-lists from features_to_integers  
  - Dictionary of visits and their LOS + outcome indicators from features_to_integers  

Output:
  - A Tuple, (trimmed list-of-lists, list of labels)

Steps:

  - Using dictionary and global settings, compute appropriate lookback period (tp.find_cutpoints)
  - Trim sequences to the appropriate length determined, combine with labels to form tuple

### 4. Keras modelling / Hyperparameter tuning  

Either python/model.py or python/hp_tune.py  

Splits data into train, validation, and test generators and converts to RaggedTensor representation. This is then fed into the keras model as defined in the script and the metrics evaluated.  
