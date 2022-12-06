"""
Keras models
DAN, LSTM, HP-tuned DAN + LSTM
"""
import argparse
import csv
import os
import pickle as pkl

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import TensorBoard

import tools.analysis as ta
import tools.preprocessing as tp
import tools.keras as tk

from azureml.core import Workspace, Dataset
from azureml.core import Run
import mlflow


class LogToAzure(keras.callbacks.Callback):
    '''Keras Callback for realtime logging to Azure'''
    def __init__(self, run):
        super(LogToAzure, self).__init__()
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        # Log all log data to Azure
        for k, v in logs.items():
            self.run.log(k, v)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model",
                        type=str,
                        default="dan",
                        choices=["dan", "lstm", "hp_lstm", "hp_dan"],
                        help="Type of Keras model to use")
    parser.add_argument("--max_seq",
                        type=int,
                        default=225,
                        help="max number of days to include")
    parser.add_argument("--outcome",
                        type=str,
                        default="misa_pt",
                        choices=["misa_pt", "multi_class", "death", "icu"],
                        help="which outcome to use as the prediction target")
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
    parser.add_argument("--demog",
                        type=bool,
                        default=True,
                        help="Should the model include patient demographics?")
    parser.add_argument('--stratify',
                        type=str,
                        default='all',
                        choices=['all', 'death', 'misa_pt', 'icu'],
                        help='which label to use for the train-test split')
    parser.add_argument('--cohort_prefix',
                        type=str,
                        default='',
                        help='prefix for the cohort csv file, ending with _s')
    parser.add_argument("--dropout",
                        type=float,
                        default=0.0,
                        help="Amount of dropout to apply")
    parser.add_argument("--recurrent_dropout",
                        type=float,
                        default=0.0,
                        help="Amount of recurrent dropout (if LSTM)")
    parser.add_argument("--n_cells",
                        type=int,
                        default=128,
                        help="Number of cells in the hidden layer")
    parser.add_argument("--batch_size",
                        type=int,
                        default=64,
                        help="Mini batch size")
    parser.add_argument("--weighted_loss",
                        help="Weight loss to account for class imbalance",
                        dest='weighted_loss',
                        action='store_true')
    parser.set_defaults(weighted_loss=False)
    parser.add_argument("--epochs",
                        type=int,
                        default=20,
                        help="Maximum epochs to run")
    parser.add_argument("--out_dir",
                        type=str,
                        help="output directory (optional)")
    parser.add_argument("--data_dir",
                        type=str,
                        help="path to the Premier data (optional)")
    parser.add_argument("--test_split",
                        type=float,
                        default=0.2,
                        help="Percentage of total data to use for testing")
    parser.add_argument("--validation_split",
                        type=float,
                        default=0.2,
                        help="Percentage of train data to use for validation")
    parser.add_argument("--rand_seed", type=int, default=2021, help="RNG seed")
    parser.add_argument(
        "--tb_update_freq",
        type=int,
        default=100,
        help="How frequently (in batches) should Tensorboard write diagnostics?"
    )

    # Parse Args, assign to globals
    args = parser.parse_args()

    TIME_SEQ = args.max_seq
    MOD_NAME = args.model
    WEIGHTED_LOSS = args.weighted_loss
    if WEIGHTED_LOSS:
        MOD_NAME += '_w'
    OUTCOME = args.outcome
    DEMOG = args.demog
    CHRT_PRFX = args.cohort_prefix
    STRATIFY = args.stratify
    DAY_ONE_ONLY = args.day_one
    if DAY_ONE_ONLY and ('lstm' not in MOD_NAME):
        # Optionally limiting the features to only those from the first day
        # of the actual COVID visit
        MOD_NAME += "_d1"
    LSTM_DROPOUT = args.dropout
    LSTM_RECURRENT_DROPOUT = args.recurrent_dropout
    N_LSTM = args.n_cells
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    TEST_SPLIT = args.test_split
    VAL_SPLIT = args.validation_split
    RAND = args.rand_seed
    TB_UPDATE_FREQ = args.tb_update_freq


    ########## Run AZUREML ######################
    run = Run.get_context()
    print("run name:",run.display_name)
    print("run details:",run.get_details())

    mlflow.log_param("MOD_NAME",MOD_NAME)
    mlflow.log_param("Outcome",OUTCOME)
    mlflow.log_param("DAY_ONE_ONLY",DAY_ONE_ONLY)
    mlflow.log_param("DEMOG",DEMOG)
    mlflow.log_param("BATCH_SIZE",BATCH_SIZE)
    mlflow.log_param("EPOCHS",EPOCHS)

    ws = run.experiment.workspace
    data_store = ws.get_default_datastore()

       ##########Loading the data from datastore
    print("Creating dataset from Datastore")
    inputs = Dataset.File.from_files(path=data_store.path('output/pkl/trimmed_seqs.pkl'))
    vocab = Dataset.File.from_files(path=data_store.path('output/pkl/all_ftrs_dict.pkl'))
    all_feats = Dataset.File.from_files(path=data_store.path('output/pkl/feature_lookup.pkl'))
    demog_dict = Dataset.File.from_files(path=data_store.path('output/pkl/demog_dict.pkl'))
    cohort = Dataset.Tabular.from_delimited_files(path=data_store.path('output/cohort/cohort.csv'))


    # DIRS
    pwd = os.path.dirname(__file__)

    # If no args are passed to overwrite these values, use repo structure to construct
    #data_dir = os.path.abspath(os.path.join(pwd, "..", "data", "data", ""))
    #output_dir = os.path.abspath(os.path.join(pwd, "..","output/", ""))

    data_dir = os.path.abspath(os.path.join(pwd,"data", "data", ""))
    output_dir = os.path.abspath(os.path.join(pwd,"output"))

    if args.data_dir is not None:
        data_dir = os.path.abspath(args.data_dir)

    if args.out_dir is not None:
        output_dir = os.path.abspath(args.out_dir)

    tensorboard_dir = os.path.abspath(
        os.path.join(data_dir, "..", "model_checkpoints"))
    pkl_dir = os.path.join(output_dir, "pkl")
    stats_dir = os.path.join(output_dir, "analysis")
    probs_dir = os.path.join(stats_dir, "probs")
    cohort_dir = os.path.join(output_dir, "cohort")
    model_outputs_dir = os.path.join("outputs", "model")

    # Create analysis dir if it doesn't exist
    [
        os.makedirs(directory, exist_ok=True)
        for directory in [stats_dir, probs_dir,tensorboard_dir,pkl_dir,cohort_dir,model_outputs_dir]
    ]

    # FILES Created
    stats_file = os.path.join(stats_dir, OUTCOME + "_stats.csv")
    probs_file = os.path.join(probs_dir, MOD_NAME + "_" + OUTCOME + ".pkl")
    preds_file = os.path.join(stats_dir, OUTCOME + "_preds.csv")


    ########################## Download data in pkl dir
    inputs.download(target_path=pkl_dir,overwrite=True,ignore_not_found=True)
    vocab.download(target_path=pkl_dir,overwrite=True,ignore_not_found=True)
    all_feats.download(target_path=pkl_dir,overwrite=True,ignore_not_found=True)
    demog_dict.download(target_path=pkl_dir,overwrite=True,ignore_not_found=True)
    cohort.to_pandas_dataframe().to_csv(os.path.join(cohort_dir,'cohort.csv'))

    # Data load
    with open(os.path.join(pkl_dir, CHRT_PRFX, "trimmed_seqs.pkl"), "rb") as f:
        inputs = pkl.load(f)

    with open(os.path.join(pkl_dir, "all_ftrs_dict.pkl"), "rb") as f:
        vocab = pkl.load(f)

    with open(os.path.join(pkl_dir, "feature_lookup.pkl"), "rb") as f:
        all_feats = pkl.load(f)

    with open(os.path.join(pkl_dir, "demog_dict.pkl"), "rb") as f:
        demog_lookup = pkl.load(f)

    # Save Embedding metadata
    # We can use this with tensorboard to visualize the embeddings
    with open(os.path.join(tensorboard_dir, "emb_metadata.tsv"), "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(zip(["id"], ["word"], ["desc"]))
        writer.writerows(zip([0], ["OOV"], ["Padding/OOV"]))
        for key, value in vocab.items():
            writer.writerow([key, value, all_feats[value]])

    # Determining number of vocab entries
    N_VOCAB = len(vocab) + 1
    N_DEMOG = len(demog_lookup) + 1
    MAX_DEMOG = max(len(x) for _, x, _ in inputs)

    # Setting y here so it's stable
    cohort = pd.read_csv(os.path.join(output_dir, CHRT_PRFX, 'cohort','cohort.csv'))
    labels = cohort[OUTCOME]
    y = cohort[OUTCOME].values.astype(np.uint8)

    N_CLASS = y.max() + 1

    inputs = [[l for l in x] for x in inputs]

    for i, x in enumerate(inputs):
        x[2] = y[i]

    # Create some metrics
    metrics = [
        keras.metrics.AUC(num_thresholds=int(1e5), name="ROC-AUC"),
        keras.metrics.AUC(num_thresholds=int(1e5), curve="PR", name="PR-AUC"),
    ]

    # Define loss function
    # NOTE: We were experimenting with focal loss at one point, maybe we can try that again at some point
    if OUTCOME == 'multi_class':
        loss_fn = keras.losses.categorical_crossentropy
    else:
        loss_fn = keras.losses.binary_crossentropy

    # Splitting the data; 'all' will produce the same test sample
    # for every outcome (kinda nice)
    if STRATIFY == 'all':
        outcomes = ['icu', 'misa_pt', 'death']
        strat_var = cohort[outcomes].values.astype(np.uint8)
    else:
        strat_var = y

    train, test = train_test_split(range(len(inputs)),
                                   test_size=TEST_SPLIT,
                                   stratify=strat_var,
                                   random_state=RAND)

    train, val = train_test_split(train,
                                  test_size=VAL_SPLIT,
                                  stratify=strat_var[train],
                                  random_state=RAND)

    # Optional weighting
    if WEIGHTED_LOSS:
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y[train])
        weight_dict = {c: weights[i] for i, c in enumerate(classes)}
    else:
        weight_dict = None

# Create some callbacks
    callbacks = [
        TensorBoard(
            log_dir=os.path.join(tensorboard_dir, OUTCOME, MOD_NAME),
            histogram_freq=1,
            update_freq=TB_UPDATE_FREQ,
            embeddings_freq=5,
            embeddings_metadata=os.path.join(tensorboard_dir,
                                             "emb_metadata.tsv"),
        ),

        # Create model checkpoint callback
        keras.callbacks.ModelCheckpoint(filepath=os.path.join(
            tensorboard_dir, OUTCOME, MOD_NAME,
            "weights.{epoch:02d}-{val_loss:.2f}.hdf5"),
                                        save_weights_only=True,
                                        monitor="val_loss",
                                        mode="max",
                                        save_best_only=True),

        # Create early stopping callback
        keras.callbacks.EarlyStopping(monitor="val_loss",
                                      min_delta=0,
                                      patience=2,
                                      mode="auto"),

        # Log epochs
        # LogToAzure(run)
    ]

    mlflow.keras.autolog(log_models=False)

    # === Long short-term memory model
    if "lstm" in MOD_NAME:
        # Produce dataset generators
        train_gen, test_gen, validation_gen = tk.create_all_data_gens(
            inputs=inputs,
            split_idx=[train, test, val],
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=RAND)

        if "hp_lstm" in MOD_NAME:
            # NOTE: IF HP-tuned, we want to use SGD with the
            # params found, so return compiled.
            model = keras.models.load_model(os.path.join(
                tensorboard_dir, "best", "lstm"),
                                            custom_objects={'tf': tf},
                                            compile=True)
        else:
            # %% Setting up the model
            model = tk.LSTM(time_seq=TIME_SEQ,
                            vocab_size=N_VOCAB,
                            n_classes=N_CLASS,
                            n_demog=N_DEMOG,
                            n_demog_bags=MAX_DEMOG,
                            ragged=True,
                            lstm_dropout=LSTM_DROPOUT,
                            recurrent_dropout=LSTM_RECURRENT_DROPOUT)
            model.compile(optimizer="adam", loss=loss_fn, metrics=metrics)

        # Train
        fitting = model.fit(train_gen,
                            validation_data=validation_gen,
                            epochs=EPOCHS,
                            callbacks=callbacks,
                            class_weight=weight_dict)

        # Produce validation and test predictions
        val_probs = model.predict(validation_gen)
        test_probs = model.predict(test_gen)

    # === Deep Averaging Network
    elif "dan" in MOD_NAME:
        if DAY_ONE_ONLY:
            # Optionally limiting the features to only those from the first day
            # of the actual COVID visit
            features = [l[0][-1] for l in inputs]
        else:
            features = [tp.flatten(l[0]) for l in inputs]

        # Optionally mixing in the demographic features
        if DEMOG:
            new_demog = [[i + N_VOCAB - 1 for i in l[1]] for l in inputs]
            features = [
                features[i] + new_demog[i] for i in range(len(features))
            ]
            demog_vocab = {k: v + N_VOCAB - 1 for k, v in demog_lookup.items()}
            vocab.update(demog_vocab)
            N_VOCAB = np.max([np.max(l) for l in features]) + 1

        # Making the variables
        X = keras.preprocessing.sequence.pad_sequences(features,
                                                       padding='post')

        # DAN Model feeds in all features at once, so there's no need to limit to the
        # sequence length, which is a size of time steps. Here we take the maximum size
        # of features that pad_sequences padded all the samples up to in the previous step.
        print("X shape:",X.shape)

        TIME_SEQ = X.shape[1]

        print("maximum feature seq: ",TIME_SEQ)

        if "hp_dan" in MOD_NAME:
            # NOTE: IF HP-tuned, we want to use SGD with the
            # params found, so return compiled.
            # HACK: This kind of assumes we're tuning for multiclass,
            # and I'm not really sure a way around that.
            n_values = np.max(y) + 1
            y_one_hot = np.eye(n_values)[y]

            model = keras.models.load_model(os.path.join(
                tensorboard_dir, "best", "dan"),
                                            custom_objects={'tf': tf},
                                            compile=True)

            model.fit(X[train],
                      y_one_hot[train],
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      validation_data=(X[val], y_one_hot[val]),
                      callbacks=callbacks,
                      class_weight=weight_dict)

        # Handle multiclass case
        elif N_CLASS > 2:
            # We have to pass one-hot labels for model fit, but CLF metrics
            # will take indices
            y_one_hot = ta.onehot_matrix(y)

            # Produce DAN model to fit
            model = tk.DAN(vocab_size=N_VOCAB,
                           ragged=False,
                           input_length=TIME_SEQ,
                           n_classes=N_CLASS)

            model.compile(optimizer="adam", loss=loss_fn, metrics=metrics)

            model.fit(X[train],
                      y_one_hot[train],
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      validation_data=(X[val], y_one_hot[val]),
                      callbacks=callbacks,
                      class_weight=weight_dict)
        else:
            # Produce DAN model to fit
            model = tk.DAN(vocab_size=N_VOCAB,
                           ragged=False,
                           input_length=TIME_SEQ)

            model.compile(optimizer="adam", loss=loss_fn, metrics=metrics)

            model.fit(X[train],
                      y[train],
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      validation_data=(X[val], y[val]),
                      callbacks=callbacks,
                      class_weight=weight_dict)

        # Produce DAN predictions on validation and test sets
        val_probs = model.predict(X[val])
        test_probs = model.predict(X[test])

    # === All model metrics, preds, and probs handling
    if N_CLASS <= 2:
        # If we are in the binary case, compute grid metrics on validation data
        # and compute the cutpoint for the test set.
        val_gm = ta.grid_metrics(y[val], val_probs)

        # Computed threshold cutpoint based on F1
        # NOTE: we could change that too. Maybe that's not the best objective
        cutpoint = val_gm.cutoff.values[np.argmax(val_gm.f1)]

        # Getting the stats
        stats = ta.clf_metrics(y[test],
                               test_probs,
                               cutpoint=cutpoint,
                               mod_name=MOD_NAME)
        mlflow.log_param("f1-score",stats.to_dict()['f1'][0])
        mlflow.log_param("auc",stats.to_dict()['auc'][0])

        # Creating probability dict to save
        prob_out = {'cutpoint': cutpoint, 'probs': test_probs}

        # Getting the test predictions
        test_preds = ta.threshold(test_probs, cutpoint)

    else:
        # Getting the stats
        stats = ta.clf_metrics(y[test],
                               test_probs,
                               average="weighted",
                               mod_name=MOD_NAME)
        mlflow.log_metric("f1-score",stats.to_dict()['f1'][0])
        mlflow.log_metric("auc",stats.to_dict()['auc'][0])

        # Creating probability dict to save
        prob_out = {'cutpoint': 0.5, 'probs': test_probs}

        # Getting the test predictions
        test_preds = np.argmax(test_probs, axis=1)

        # Saving the max from each row for writing to CSV
        test_probs = np.amax(test_probs, axis=1)

    # --- Writing the metrics results to disk
    # Optionally append results if file already exists
    append_file = os.path.exists(stats_file)

    stats.to_csv(stats_file,
                 mode="a" if append_file else "w",
                 header=False if append_file else True,
                 index=False)

    # --- Writing the predicted probabilities to disk
    with open(probs_file, "wb") as f:
        pkl.dump(prob_out, f)

    ##### SAVING model in azure outputs folder
    model.save(os.path.join(model_outputs_dir,MOD_NAME + "_" + OUTCOME +".h5"))
    # Signature
    if 'dan' in MOD_NAME:
        signature = mlflow.models.infer_signature(X[test],test_preds)
        mlflow.keras.log_model(model,
                            model_outputs_dir,
                            signature=signature)
    if 'lstm' in MOD_NAME:
        mlflow.keras.log_model(model,
                            model_outputs_dir)

    # --- Writing the test predictions to the test predictions CSV

    #if os.path.exists(preds_file):
    #    preds_df = pd.read_csv(preds_file)
    #else:
    #    preds_df = pd.read_csv(
    #        os.path.join(output_dir, OUTCOME + "_cohort.csv"))
    #    preds_df = preds_df.iloc[test, :]

    #preds_df[MOD_NAME + '_prob'] = test_probs
    #preds_df[MOD_NAME + '_pred'] = test_preds
    #preds_df.to_csv(preds_file, index=False)
