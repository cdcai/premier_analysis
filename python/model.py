"""
Starter Keras model
"""
# %%
import csv
import os
import pickle as pkl
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from focal_loss import BinaryFocalLoss
from sklearn.metrics import (average_precision_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import TensorBoard

import tools.analysis as ta
from tools import keras as tk
from tools.analysis import grid_metrics

# %% Globals
TIME_SEQ = 225
TARGET = "misa_pt"
RAGGED = True
LSTM_DROPOUT = 0.4
# NOTE: Recurrent dropout is advisable, but it also means
# you forgoe CuDNN-optimization for the LSTM, so it will train
# about 1/3 slower on GPU
LSTM_RECURRENT_DROPOUT = 0.4
N_LSTM = 128
# NOTE: I maxed out my GPU running 32, 20 ran ~7.8GB on GPU
BATCH_SIZE = 32
EPOCHS = 20
# NOTE: Take only a small sample of the data to fit?
SUBSAMPLE = False
SAMPLE_FRAC = 0.05
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1
RAND = 2020
TB_UPDATE_FREQ = 100
# %% Load in Data
output_dir = os.path.abspath("../output/") + "/"
tensorboard_dir = os.path.abspath("../data/model_checkpoints/") + "/"
data_dir = os.path.abspath("../data/data/") + "/"
pkl_dir = output_dir + "pkl/"

with open(pkl_dir + "trimmed_seqs.pkl", "rb") as f:
    inputs = pkl.load(f)

with open(pkl_dir + "all_ftrs_dict.pkl", "rb") as f:
    vocab = pkl.load(f)

with open(pkl_dir + "feature_lookup.pkl", "rb") as f:
    all_feats = pkl.load(f)

# %% Save Embedding metadata
# We can use this with tensorboard to visualize the embeddings
with open(tensorboard_dir + "emb_metadata.tsv", "w") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(zip(["id"], ["word"], ["desc"]))
    writer.writerows(zip([0], ["OOV"], ["Padding/OOV"]))
    for key, value in vocab.items():
        writer.writerow([key, value, all_feats[value]])

# %% Determining number of vocab entries
N_VOCAB = len(vocab) + 1

# %% Subsampling if desired
if SUBSAMPLE:
    _, inputs, _, _ = train_test_split(
        inputs,
        [labs for _, labs in inputs],
        test_size=SAMPLE_FRAC,
        random_state=RAND,
        stratify=[labs for _, labs in inputs],
    )

# %% Split into test/train
train, test, _, _ = train_test_split(
    inputs,
    [labs for _, labs in inputs],
    test_size=TEST_SPLIT,
    random_state=RAND,
    stratify=[labs for _, labs in inputs],
)

# Further split into train/validation
train, validation, _, _ = train_test_split(
    train,
    [labs for _, labs in train],
    test_size=VAL_SPLIT,
    random_state=RAND,
    stratify=[labs for _, labs in train],
)

# %% Compute steps-per-epoch
# NOTE: Sometimes it can't determine this properly from tf.data
STEPS_PER_EPOCH = len(train) // BATCH_SIZE
VALID_STEPS_PER_EPOCH = len(validation) // BATCH_SIZE

# %% Compute class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique([labs for _, labs in train]),
    y=[labs for _, labs in train],
)

class_weights = dict(zip([0, 1], class_weights))

# %% Compute initial output bias
neg, pos = np.bincount([lab for _, lab in train])
out_bias = np.log([pos / neg])

# %%
train_gen = tk.create_ragged_data(train,
                                  max_time=TIME_SEQ,
                                  epochs=EPOCHS,
                                  random_seed=RAND,
                                  resample=False,
                                  resample_frac=[0.9, 0.1],
                                  batch_size=BATCH_SIZE)

validation_gen = tk.create_ragged_data(validation,
                                       max_time=TIME_SEQ,
                                       epochs=EPOCHS,
                                       random_seed=RAND,
                                       batch_size=BATCH_SIZE)

# NOTE: don't shuffle test data
test_gen = tk.create_ragged_data(test,
                                 max_time=TIME_SEQ,
                                 epochs=1,
                                 shuffle=False,
                                 random_seed=RAND,
                                 batch_size=BATCH_SIZE)

# SEtting up the model
model = tk.LSTM(time_seq=TIME_SEQ,
                vocab_size=N_VOCAB,
                ragged=RAGGED,
                output_bias=out_bias,
                lstm_dropout=LSTM_DROPOUT,
                recurrent_dropout=LSTM_RECURRENT_DROPOUT,
                batch_size=BATCH_SIZE)

model.compile(
    optimizer="adam",
    # NOTE: TFA focal loss is failing sporadically. I think it has something
    # to do with regularization, but I've not been able to consistently reproduce.
    # the focal-loss implementation doesn't seem to have these issues.
    loss=BinaryFocalLoss(gamma=2.0, pos_weight=0.25),
    metrics=[
        keras.metrics.AUC(num_thresholds=int(1e5), name="ROC-AUC"),
        keras.metrics.AUC(num_thresholds=int(1e5), curve="PR", name="PR-AUC"),
    ])

model.summary()

# Create Tensorboard callback
tb_callback = TensorBoard(
    log_dir=tensorboard_dir + "/" + TARGET + "/" +
    datetime.now().strftime("%Y%m%d-%H%M%S") + "/",
    histogram_freq=1,
    update_freq=TB_UPDATE_FREQ,
    embeddings_freq=5,
    embeddings_metadata=output_dir + "emb_metadata.tsv",
)

# Create model checkpoint callback
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=tensorboard_dir + "/" + TARGET + "/" +
    "weights.{epoch:02d}-{val_loss:.2f}.hdf5",
    save_weights_only=True,
    monitor="val_loss",
    mode="max",
    save_best_only=True)

# Create early stopping callback
stopping_checkpoint = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                    min_delta=0,
                                                    patience=2,
                                                    mode="auto")

# %% Train
fitting = model.fit(
    train_gen,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_gen,
    validation_steps=VALID_STEPS_PER_EPOCH,
    epochs=EPOCHS,
    callbacks=[tb_callback, model_checkpoint_callback, stopping_checkpoint])

# Test
print(model.evaluate(test_gen))

# %% Validation F1 cut
y_validation = [[(lambda x: [0] if x == [] else x)(bags) for bags in seq]
                for seq, _ in validation]

y_validation = tf.ragged.constant(y_validation)

y_pred_validation = model.predict(y_validation)

y_true_validation = [lab for _, lab in validation]

# Resizing for output which is divisible by BATCH_SIZE
y_true_validation = np.array(y_true_validation[0:y_pred_validation.shape[0]])

val_gm = ta.grid_metrics(y_true_validation,
                         y_pred_validation,
                         min=0.0,
                         max=1.0,
                         step=0.001)

f1_cut = val_gm.cutoff.values[np.argmax(val_gm.f1)]
# %% Predicting on test data
y_pred_test = model.predict(test_gen)

y_true_test = [lab for _, lab in test]
y_true_test = np.array(y_true_test[0:y_pred_test.shape[0]])

# %% Print the stats when taking the cutpoint from the validation set (not cheating)
lstm_stats = ta.clf_metrics(y_true_test, ta.threshold(y_pred_test, f1_cut))
auc = roc_auc_score(y_true_test, y_pred_test)
pr = average_precision_score(y_true_test, y_pred_test)
lstm_stats['auc'] = auc
lstm_stats['ap'] = pr

print(lstm_stats)
# %% Run grid metrics on test anyways just to see overall
output = grid_metrics(y_true_test, y_pred_test)
print(output.sort_values("f1"))

output.to_csv(tensorboard_dir + "/" + TARGET + "/" + "grid_metrics.csv",
              index=False)
print("ROC-AUC: {}".format(roc_auc_score(y_true_test, y_pred_test)))
