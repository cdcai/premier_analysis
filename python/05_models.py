# Databricks notebook source
# MAGIC %pip install keras-tuner --quiet
# MAGIC %pip install mlflow --quiet

# COMMAND ----------

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

import mlflow

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text(
  name='experiment_id',
  defaultValue='1645704359340635',
  label='Experiment ID'
)

# COMMAND ----------

dbutils.widgets.dropdown("model", "dan", ["dan", "lstm", "hp_lstm", "hp_dan"])
MOD_NAME = dbutils.widgets.get("model")

# COMMAND ----------

dbutils.widgets.dropdown("outcome","icu",["misa_pt", "multi_class", "death", "icu"])
OUTCOME = dbutils.widgets.get("outcome")

# COMMAND ----------

dbutils.widgets.dropdown("demographics", "True", ["True", "False"])
DEMOG = dbutils.widgets.get("demographics")
if DEMOG == "True": DEMOG = True
else: DEMOG = False

# COMMAND ----------

dbutils.widgets.dropdown("stratify", "all", ['all', 'death', 'misa_pt', 'icu'])
STRATIFY = dbutils.widgets.get("stratify")

# COMMAND ----------

import mlflow
experiment = dbutils.widgets.get("experiment_id")
assert experiment is not None
current_experiment = mlflow.get_experiment(experiment)
assert current_experiment is not None
experiment_id= current_experiment.experiment_id
mlflow.set_experiment(experiment_id=experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ```
# MAGIC parser = argparse.ArgumentParser()
# MAGIC 
# MAGIC     parser.add_argument("--model",
# MAGIC                         type=str,
# MAGIC                         default="dan",
# MAGIC                         choices=["dan", "lstm", "hp_lstm", "hp_dan"],
# MAGIC                         help="Type of Keras model to use")
# MAGIC     parser.add_argument("--max_seq",
# MAGIC                         type=int,
# MAGIC                         default=225,
# MAGIC                         help="max number of days to include")
# MAGIC     parser.add_argument("--outcome",
# MAGIC                         type=str,
# MAGIC                         default="misa_pt",
# MAGIC                         choices=["misa_pt", "multi_class", "death", "icu"],
# MAGIC                         help="which outcome to use as the prediction target")
# MAGIC     parser.add_argument(
# MAGIC         '--day_one',
# MAGIC         help="Use only first inpatient day's worth of features (DAN only)",
# MAGIC         dest='day_one',
# MAGIC         action='store_true')
# MAGIC     parser.add_argument('--all_days',
# MAGIC                         help="Use all features in lookback period (DAN only)",
# MAGIC                         dest='day_one',
# MAGIC                         action='store_false')
# MAGIC     parser.set_defaults(day_one=True)
# MAGIC     parser.add_argument("--demog",
# MAGIC                         type=bool,
# MAGIC                         default=True,
# MAGIC                         help="Should the model include patient demographics?")
# MAGIC     parser.add_argument('--stratify',
# MAGIC                         type=str,
# MAGIC                         default='all',
# MAGIC                         choices=['all', 'death', 'misa_pt', 'icu'],
# MAGIC                         help='which label to use for the train-test split')
# MAGIC     parser.add_argument('--cohort_prefix',
# MAGIC                         type=str,
# MAGIC                         default='',
# MAGIC                         help='prefix for the cohort csv file, ending with _s')
# MAGIC     parser.add_argument("--dropout",
# MAGIC                         type=float,
# MAGIC                         default=0.0,
# MAGIC                         help="Amount of dropout to apply")
# MAGIC     parser.add_argument("--recurrent_dropout",
# MAGIC                         type=float,
# MAGIC                         default=0.0,
# MAGIC                         help="Amount of recurrent dropout (if LSTM)")
# MAGIC     parser.add_argument("--n_cells",
# MAGIC                         type=int,
# MAGIC                         default=128,
# MAGIC                         help="Number of cells in the hidden layer")
# MAGIC     parser.add_argument("--batch_size",
# MAGIC                         type=int,
# MAGIC                         default=64,
# MAGIC                         help="Mini batch size")
# MAGIC     parser.add_argument("--weighted_loss",
# MAGIC                         help="Weight loss to account for class imbalance",
# MAGIC                         dest='weighted_loss',
# MAGIC                         action='store_true')
# MAGIC     parser.set_defaults(weighted_loss=False)
# MAGIC     parser.add_argument("--epochs",
# MAGIC                         type=int,
# MAGIC                         default=20,
# MAGIC                         help="Maximum epochs to run")
# MAGIC     parser.add_argument("--out_dir",
# MAGIC                         type=str,
# MAGIC                         help="output directory (optional)")
# MAGIC     parser.add_argument("--data_dir",
# MAGIC                         type=str,
# MAGIC                         help="path to the Premier data (optional)")
# MAGIC     parser.add_argument("--test_split",
# MAGIC                         type=float,
# MAGIC                         default=0.2,
# MAGIC                         help="Percentage of total data to use for testing")
# MAGIC     parser.add_argument("--validation_split",
# MAGIC                         type=float,
# MAGIC                         default=0.2,
# MAGIC                         help="Percentage of train data to use for validation")
# MAGIC     parser.add_argument("--rand_seed", type=int, default=2021, help="RNG seed")
# MAGIC     parser.add_argument(
# MAGIC         "--tb_update_freq",
# MAGIC         type=int,
# MAGIC         default=100,
# MAGIC         help="How frequently (in batches) should Tensorboard write diagnostics?"
# MAGIC     )
# MAGIC 
# MAGIC     # Parse Args, assign to globals
# MAGIC     args = parser.parse_args()
# MAGIC 
# MAGIC     TIME_SEQ = args.max_seq
# MAGIC     MOD_NAME = args.model
# MAGIC     WEIGHTED_LOSS = args.weighted_loss
# MAGIC     if WEIGHTED_LOSS:
# MAGIC         MOD_NAME += '_w'
# MAGIC     OUTCOME = args.outcome
# MAGIC     DEMOG = args.demog
# MAGIC     CHRT_PRFX = args.cohort_prefix
# MAGIC     STRATIFY = args.stratify
# MAGIC     DAY_ONE_ONLY = args.day_one
# MAGIC     if DAY_ONE_ONLY and ('lstm' not in MOD_NAME):
# MAGIC         # Optionally limiting the features to only those from the first day
# MAGIC         # of the actual COVID visit
# MAGIC         MOD_NAME += "_d1"
# MAGIC     LSTM_DROPOUT = args.dropout
# MAGIC     LSTM_RECURRENT_DROPOUT = args.recurrent_dropout
# MAGIC     N_LSTM = args.n_cells
# MAGIC     BATCH_SIZE = args.batch_size
# MAGIC     EPOCHS = args.epochs
# MAGIC     TEST_SPLIT = args.test_split
# MAGIC     VAL_SPLIT = args.validation_split
# MAGIC     RAND = args.rand_seed
# MAGIC     TB_UPDATE_FREQ = args.tb_update_freq
# MAGIC 
# MAGIC     # DIRS
# MAGIC     pwd = os.path.dirname(__file__)
# MAGIC 
# MAGIC     # If no args are passed to overwrite these values, use repo structure to construct
# MAGIC     data_dir = os.path.abspath(os.path.join(pwd, "..", "data", "data", ""))
# MAGIC     output_dir = os.path.abspath(os.path.join(pwd, "output/", ""))
# MAGIC 
# MAGIC     if args.data_dir is not None:
# MAGIC         data_dir = os.path.abspath(args.data_dir)
# MAGIC 
# MAGIC     if args.out_dir is not None:
# MAGIC         output_dir = os.path.abspath(args.out_dir)
# MAGIC 
# MAGIC     tensorboard_dir = os.path.abspath(
# MAGIC         os.path.join(data_dir, "..", "model_checkpoints"))
# MAGIC     pkl_dir = os.path.join(output_dir, "pkl")
# MAGIC     stats_dir = os.path.join(output_dir, "analysis")
# MAGIC     probs_dir = os.path.join(stats_dir, "probs")
# MAGIC 
# MAGIC     # Create analysis dir if it doesn't exist
# MAGIC     [
# MAGIC         os.makedirs(directory, exist_ok=True)
# MAGIC         for directory in [stats_dir, probs_dir]
# MAGIC     ]
# MAGIC 
# MAGIC     # FILES Created
# MAGIC     stats_file = os.path.join(stats_dir, OUTCOME + "_stats.csv")
# MAGIC     probs_file = os.path.join(probs_dir, MOD_NAME + "_" + OUTCOME + ".pkl")
# MAGIC     preds_file = os.path.join(stats_dir, OUTCOME + "_preds.csv")
# MAGIC ```

# COMMAND ----------

TIME_SEQ = 225
#MOD_NAME = 'dan'   #'lstm'
WEIGHTED_LOSS = True
if WEIGHTED_LOSS:
    MOD_NAME += '_w'
#OUTCOME = 'misa_pt'
#DEMOG = True
CHRT_PRFX = 'dan'
#STRATIFY = 'all'
DAY_ONE_ONLY = True
if DAY_ONE_ONLY and ('lstm' not in MOD_NAME):
    # Optionally limiting the features to only those from the first day
    # of the actual COVID visit
    MOD_NAME += "_d1"
LSTM_DROPOUT = 0.00
LSTM_RECURRENT_DROPOUT = 0.00
N_LSTM = 16  #128
BATCH_SIZE = 128  # adjusting due to memory error
EPOCHS = 10
TEST_SPLIT = 0.20
VAL_SPLIT = 0.10
RAND = 2022
TB_UPDATE_FREQ = 100

# If no args are passed to overwrite these values, use repo structure to construct
output_dir = '/dbfs/home/tnk6/premier_output/'
data_dir = '/dbfs/home/tnk6/premier/'

if data_dir is not None:
    data_dir = os.path.abspath(data_dir)

if output_dir is not None:
    output_dir = os.path.abspath(output_dir)

tensorboard_dir = os.path.abspath(
    os.path.join(output_dir, "model_checkpoints"))
pkl_dir = os.path.join(output_dir, "pkl")
stats_dir = os.path.join(output_dir, "analysis")
probs_dir = os.path.join(stats_dir, "probs")

# Create analysis dir if it doesn't exist
[
    os.makedirs(directory, exist_ok=True)
    for directory in [stats_dir, probs_dir, tensorboard_dir]
]

# FILES Created
stats_file = os.path.join(stats_dir, OUTCOME + "_stats.csv")
probs_file = os.path.join(probs_dir, MOD_NAME + "_" + OUTCOME + ".pkl")
preds_file = os.path.join(stats_dir, OUTCOME + "_preds.csv")

# COMMAND ----------

# Data load
with open(os.path.join(pkl_dir, "trimmed_seqs.pkl"), "rb") as f:
    inputs = pkl.load(f)

with open(os.path.join(pkl_dir, "all_ftrs_dict.pkl"), "rb") as f:
    vocab = pkl.load(f)

with open(os.path.join(pkl_dir, "feature_lookup.pkl"), "rb") as f:
    all_feats = pkl.load(f)

with open(os.path.join(pkl_dir, "demog_dict.pkl"), "rb") as f:
    demog_lookup = pkl.load(f)


# COMMAND ----------

# Save Embedding metadata
# We can use this with tensorboard to visualize the embeddings
with open(os.path.join(tensorboard_dir, "emb_metadata.tsv"), "w") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(zip(["id"], ["word"], ["desc"]))
    writer.writerows(zip([0], ["OOV"], ["Padding/OOV"]))
    for key, value in vocab.items():
        writer.writerow([key, value, all_feats[value]])

# COMMAND ----------


#
# add execution parameters to MLFLOW
#

mlflow.end_run()
mlflow.start_run(experiment_id=experiment_id)
mlflow.autolog()
mlflow.log_param("demographics", DEMOG)
mlflow.log_param("outcome", OUTCOME)
mlflow.log_param("stratify", STRATIFY)

# Determining number of vocab entries
N_VOCAB = len(vocab) + 1
N_DEMOG = len(demog_lookup) + 1
MAX_DEMOG = max(len(x) for _, x, _ in inputs)

# Setting y here so it's stable
cohort = pd.read_csv(os.path.join(output_dir, 'cohort.csv'))
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
                                  mode="auto")
]

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
    TIME_SEQ = X.shape[1]

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
              # callbacks=callbacks,
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
        print('Starting DAN model')
        
        model.fit(X[train],
                  y[train],
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_data=(X[val], y[val]),
                  #callbacks=callbacks,
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

    # Creating probability dict to save
    prob_out = {'cutpoint': 0.5, 'probs': test_probs}

    # Getting the test predictions
    test_preds = np.argmax(test_probs, axis=1)

    # Saving the max from each row for writing to CSV
    test_probs = np.amax(test_probs, axis=1)
#
#
# add metrics to MLFLow
#
mlflow.keras.log_model(model, "model")
for i in stats.columns:
    if not isinstance(stats[i].iloc[0], str):
        mlflow.log_metric("testing_"+i, stats[i].iloc[0])
#
# END MLFLOW RUN
#
mlflow.end_run()


# COMMAND ----------


