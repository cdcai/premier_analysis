"""
Hyperparameter tuning Keras models with kerastuner
"""
import csv
import os
import pickle as pkl
from datetime import datetime

import kerastuner
import kerastuner.tuners as tuners
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import TensorBoard

from tools import keras as tk
from tools.analysis import grid_metrics

# === Globals
TIME_SEQ = 225
TARGET = "hp_tuned"
RAGGED = True
BATCH_SIZE = 32
EPOCHS = 20
# NOTE: Take only a small sample of the data to fit?
SUBSAMPLE = False
SAMPLE_FRAC = 0.1
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1
RAND = 2020
TB_UPDATE_FREQ = 1000

# === Paths
output_dir = os.path.abspath("output/") + "/"
tensorboard_dir = os.path.abspath("data/model_checkpoints/") + "/"
data_dir = os.path.abspath("data/data/") + "/"
pkl_dir = output_dir + "pkl/"

# === Load in data
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
        writer.writerow([value, key, all_feats[key]])
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

# %% Compute class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique([labs for _, labs in train]),
    y=[labs for _, labs in train],
)

class_weights = dict(zip([0, 1], class_weights))

# %% Compute steps-per-epoch
# NOTE: Sometimes it can't determine this properly from tf.data
STEPS_PER_EPOCH = len(train) // BATCH_SIZE
VALID_STEPS_PER_EPOCH = len(validation) // BATCH_SIZE

# %% Compute initial output bias
neg, pos = np.bincount([lab for _, lab in train])

out_bias = np.log([pos / neg])
# %%
train_gen = tk.create_ragged_data(train,
                                  max_time=TIME_SEQ,
                                  epochs=EPOCHS,
                                  random_seed=RAND,
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

# %% Generate Hyperparameter model
hyper_model = tk.LSTMHyperModel(ragged=RAGGED,
                                n_timesteps=TIME_SEQ,
                                vocab_size=N_VOCAB,
                                batch_size=BATCH_SIZE,
                                bias_init=out_bias)

# %%
tuner = tuners.Hyperband(
    hyper_model,
    objective="val_loss",
    max_epochs=EPOCHS,
    hyperband_iterations=5,
    # loss=tfa.losses.SigmoidFocalCrossEntropy(),  # BUG: Does not run with kerastuner for some reason
    project_name="hyperparameter-tuning",
    # NOTE: This could be in output as well if we don't want to track/version it
    directory="data/model_checkpoints/",
)

# Announce the search space
tuner.search_space_summary()

# NOTE: I think this works with HParams
hparam_tb_callback = TensorBoard(log_dir=tensorboard_dir +
                                 "/hyperparameter-tuning/",
                                 histogram_freq=1,
                                 update_freq=TB_UPDATE_FREQ)

# And search the space
tuner.search(train_gen,
             validation_data=validation_gen,
             epochs=EPOCHS,
             steps_per_epoch=STEPS_PER_EPOCH,
             validation_steps=VALID_STEPS_PER_EPOCH,
             callbacks=[
                 keras.callbacks.EarlyStopping("val_loss", patience=1),
                 hparam_tb_callback
             ],
             class_weight=class_weights)

# Get results
tuner.results_summary()

# === Create model with best Hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]

best_model = tuner.hypermodel.build(best_hp)

best_model.summary()

# === Callbacks
# Create Tensorboard callback
tb_callback = TensorBoard(
    log_dir=tensorboard_dir + "/" + TARGET + "/" +
    datetime.now().strftime("%Y%m%d-%H%M%S") + "/",
    histogram_freq=1,
    update_freq=TB_UPDATE_FREQ,
    embeddings_freq=1,
    embeddings_metadata=output_dir + "emb_metadata.tsv",
)

# Create model checkpoint callback
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=tensorboard_dir + "/" + TARGET + "/" +
    "weights.{epoch:02d}-{val_loss:.2f}.hdf5",
    save_weights_only=True,
    monitor="val_loss",
    save_best_only=True)

# === Fit model
best_model.fit(train_gen,
               validation_data=validation_gen,
               epochs=EPOCHS,
               class_weight=class_weights,
               callbacks=[tb_callback, model_checkpoint_callback])

# Test
print(model.evaluate(test_gen))

# %% F1, etc
y_pred = model.predict(test_gen)

y_true = [lab for _, lab in test]

# Resizing for output which is divisible by BATCH_SIZE
y_true = np.array(y_true[0:y_pred.shape[0]])
output = grid_metrics(y_true, y_pred)
print(output)

output.to_csv(tensorboard_dir + "/" + TARGET + "/" + "grid_metrics.csv",
              index=False)
print("ROC-AUC: {}".format(roc_auc_score(y_true, y_pred)))
