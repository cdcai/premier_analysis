
import os
import time
from importlib import reload
import pandas as pd
import argparse
import numpy as np
from azureml.core import Model,Dataset
import joblib
import pickle as pkl
from azureml.core import Workspace, Experiment, Run, RunConfiguration
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split

def flatten(l):
    return [item for sublist in l for item in sublist]

def baseline_preprocessing(OUTCOME):

    DAY_ONE_ONLY = True
    USE_DEMOG = True
    TEST_SPLIT = 0.1
    VAL_SPLIT = 0.8
    RAND = 42

    run = Run.get_context()
    print("run name:",run.display_name)
    print("run details:",run.get_details())
    
    ws = run.experiment.workspace

    data_store = ws.get_default_datastore()

    print("Creating dataset from Datastore")
    inputs = Dataset.File.from_files(path=data_store.path('output/pkl/trimmed_seqs.pkl'))  
    vocab = Dataset.File.from_files(path=data_store.path('output/pkl/all_ftrs_dict.pkl'))
    demog_dict = Dataset.File.from_files(path=data_store.path('output/pkl/demog_dict.pkl'))
    cohort = Dataset.Tabular.from_delimited_files(path=data_store.path('output/cohort/cohort.csv'))
    
    pwd = os.path.dirname(__file__)
    output_dir = os.path.abspath(os.path.join(pwd,"output"))
    pkl_dir = os.path.join(output_dir, "pkl")
    csv_dir = os.path.join(output_dir, "csv")

    os.makedirs(pkl_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    print("Downloading data from Datastore...")

    inputs.download(target_path=pkl_dir,overwrite=True,ignore_not_found=True)
    vocab.download(target_path=pkl_dir,overwrite=True,ignore_not_found=True)
    demog_dict.download(target_path=pkl_dir,overwrite=True,ignore_not_found=True)
    cohort.to_pandas_dataframe().to_csv(os.path.join(csv_dir,'cohort.csv'))

    print("Loading var...")
    with open(os.path.join(pkl_dir, "trimmed_seqs.pkl"), "rb") as f:
        inputs = pkl.load(f)

    with open(os.path.join(pkl_dir, "all_ftrs_dict.pkl"), "rb") as f:
        vocab = pkl.load(f)

    with open(os.path.join(pkl_dir, "demog_dict.pkl"), "rb") as f:
        demog_dict = pkl.load(f)
        demog_dict = {k: v for v, k in demog_dict.items()}

    
    # Separating the inputs and labels
    features = [t[0] for t in inputs]
    demog = [t[1] for t in inputs]
    cohort = pd.read_csv(os.path.join(csv_dir, 'cohort.csv'))
    labels = cohort[OUTCOME]

    # Counts to use for loops and stuff
    n_patients = len(features)
    n_features = np.max(list(vocab.keys()))
    n_classes = len(np.unique(labels))
    binary = n_classes <= 2

        # Converting the labels to an array
    y = np.array(labels, dtype=np.uint8)

    # Optionally limiting the features to only those from the first day
    # of the actual COVID visit
    if DAY_ONE_ONLY:
        features = [l[-1] for l in features]
    else:
        features = [flatten(l) for l in features]

    new_demog = [[i + n_features for i in l] for l in demog]
    features = [features[i] + new_demog[i] for i in range(n_patients)]
    demog_vocab = {k + n_features: v for k, v in demog_dict.items()}
    vocab.update(demog_vocab)
    n_features = np.max([np.max(l) for l in features])
    # all_feats.update({v: v for k, v in demog_dict.items()})

    # Converting the features to a sparse matrix
    mat = lil_matrix((n_patients, n_features + 1))
    for row, cols in enumerate(features):
        mat[row, cols] = 1

    # Converting to csr because the internet said it would be faster
    print("Converting to csr..")
    X = mat.tocsr()

    # Splitting the data; 'all' will produce the same test sample
    # for every outcome (kinda nice)

    STRATIFY = None

    strat_var = y
    train, test = train_test_split(range(n_patients),
                                    test_size=TEST_SPLIT,
                                    stratify=strat_var,
                                    random_state=RAND)

    # Doing a validation split for threshold-picking on binary problems
    train, val = train_test_split(train,
                                    test_size=VAL_SPLIT,
                                    stratify=strat_var[train],
                                    random_state=RAND)

    from sklearn.decomposition import PCA,TruncatedSVD
    svd = TruncatedSVD(n_components=2000)
    
    x_train = svd.fit_transform(X[train])
    x_test = svd.transform(X[test])



    return  x_train,y[train],x_test,y[test]


if __name__ == '__main__':
    parser = argparse.ArgumentParser("feature")
    parser.add_argument("--outcome",type=str)

    args = parser.parse_args()

    OUTCOME = args.outcome

    x_train,y_train,x_test,y_test = baseline_preprocessing(OUTCOME=OUTCOME)

    #x_train_df = pd.DataFrame.sparse.from_spmatrix(x_train)
    #x_test_df = pd.DataFrame.sparse.from_spmatrix(x_test)

    #train_data = pd.concat([x_train_df,pd.DataFrame(y_train)],axis =1)
    train_data = pd.DataFrame(x_train)
    train_data['class'] = y_train
    
    print("train shape:",train_data.shape)
    
    run = Run.get_context()
    ws = run.experiment.workspace

    data_store = ws.get_default_datastore()
    pwd = os.path.dirname(__file__)
    feature_dir = os.path.join(pwd,"output","automl")
    os.makedirs(feature_dir,exist_ok=True)

    print("Saving train_data...")
    # np.savetxt(os.path.join(feature_dir,"train_data.csv"), train_data, delimiter=",")
    train_data.to_csv(os.path.join(feature_dir,"train_data.csv"),index=False)

    dataset_name = f"train-data-baseline-{OUTCOME}"
    #ds_train = Dataset.Tabular.register_pandas_dataframe(train_data,target=data_store,name=dataset_name,show_progress=True)
    data_store.upload(src_dir=feature_dir,target_path="output/automl",overwrite=True,show_progress=True)
    
    datastore_paths = [(data_store, "output/automl")]

    print("Create dataset...")
    inputs = Dataset.Tabular.from_delimited_files(path=datastore_paths)

    print("Register dataset..")
    inputs.register(name=dataset_name,workspace=ws,create_new_version=True)
