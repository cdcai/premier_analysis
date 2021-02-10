"""
Starter Keras model
"""
# %%
import csv
import os
import pickle as pkl
from datetime import datetime
import argparse

import numpy as np
import tensorflow.keras as keras
from sklearn.metrics import (roc_auc_score, average_precision_score)
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard

import tools.analysis as ta
from tools import keras as tk
from tools.analysis import grid_metrics

# %% Globals
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--outcome",
                        type=str,
                        default="dan",
                        choices=["dan", "lstm"],
                        help="Type of Keras model to use")
    parser.add_argument("--max_seq",
                        type=int,
                        default=225,
                        help="max number of days to include")
    parser.add_argument("--outcome",
                        type=str,
                        default="misa_pt",
                        choices=["misa_pt", "multi_class", "death"],
                        help="which outcome to use as the prediction target")
    parser.add_argument(
        "--day_one",
        type=bool,
        default=True,
        help=
        "(if using DAN) should we only consider features from the first inpatient day?"
    )
    parser.add_argument("--demog",
                        type=bool,
                        default=True,
                        help="Should the model include patient demographics?")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.4,
                        help="Amount of dropout to apply")
    parser.add_argument("--recurrent_dropout",
                        type=float,
                        default=0.4,
                        help="Amount of recurrent dropout (if LSTM)")
    parser.add_argument("--n_cells",
                        type=int,
                        default=128,
                        help="Number of cells in the hidden layer")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Mini batch size")
    parser.add_argument("--epochs",
                        type=int,
                        default=20,
                        help="Maximum epochs to run")
    parser.add_argument("--out_dir",
                        type=str,
                        default="output/",
                        help="output directory")
    parser.add_argument("--data_dir",
                        type=str,
                        default="data/data/",
                        help="path to the Premier data")
    parser.add_argument("--test_split",
                        type=float,
                        default=0.2,
                        help="Percentage of total data to use for testing")
    parser.add_argument("--validation_split",
                        type=float,
                        default=0.1,
                        help="Percentage of train data to use for validation")
    parser.add_argument("--rand_seed", type=int, default=2021, help="RNG seed")
    parser.add_argument(
        "--tb_update_freq",
        type=int,
        default=100,
        help="How frequently (in batches) should Tensorboard write diagnostics?"
    )
    args = parser.parse_args()

    TIME_SEQ = args.max_seq
    TARGET = args.outcome
    LSTM_DROPOUT = args.dropout
    LSTM_RECURRENT_DROPOUT = args.recurrent_dropout
    N_LSTM = args.n_cells
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    TEST_SPLIT = args.test_split
    VAL_SPLIT = args.validation_split
    RAND = args.rand_seed
    TB_UPDATE_FREQ = args.tb_update_freq

    # %% Load in Data
    output_dir = os.path.abspath(args.out_dir)
    tensorboard_dir = os.path.abspath(
        os.path.join(args.data_dir, "..", "model_checkpoints"))
    data_dir = os.path.abspath(args.data_dir)
    pkl_dir = os.path.join(output_dir, "pkl")

    with open(os.path.join(pkl_dir, "multi_class_trimmed_seqs.pkl"),
              "rb") as f:
        inputs = pkl.load(f)

    with open(os.path.join(pkl_dir, "all_ftrs_dict.pkl"), "rb") as f:
        vocab = pkl.load(f)

    with open(os.path.join(pkl_dir, "feature_lookup.pkl"), "rb") as f:
        all_feats = pkl.load(f)

    with open(os.path.join(pkl_dir, "demog_dict.pkl"), "rb") as f:
        demog_lookup = pkl.load(f)

    # %% Save Embedding metadata
    # We can use this with tensorboard to visualize the embeddings
    with open(os.path.join(tensorboard_dir, "emb_metadata.tsv"), "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(zip(["id"], ["word"], ["desc"]))
        writer.writerows(zip([0], ["OOV"], ["Padding/OOV"]))
        for key, value in vocab.items():
            writer.writerow([key, value, all_feats[value]])

    # %% Determining number of vocab entries
    N_VOCAB = len(vocab) + 1
    N_DEMOG = len(demog_lookup) + 1
    MAX_DEMOG = max(len(x) for _, x, _ in inputs)
    N_CLASS = max(x for _, _, x in inputs) + 1

    # %% Split into test/train
    train, test, _, _ = train_test_split(
        inputs,
        [labs for _, _, labs in inputs],
        test_size=TEST_SPLIT,
        random_state=RAND,
        stratify=[labs for _, _, labs in inputs],
    )

    # Further split into train/validation
    train, validation, _, _ = train_test_split(
        train,
        [labs for _, _, labs in train],
        test_size=VAL_SPLIT,
        random_state=RAND,
        stratify=[labs for _, _, labs in train],
    )

    # %% Compute steps-per-epoch
    # NOTE: Sometimes it can't determine this properly from tf.data
    STEPS_PER_EPOCH = len(train) // BATCH_SIZE
    VALID_STEPS_PER_EPOCH = len(validation) // BATCH_SIZE

    # %%
    train_gen = tk.create_ragged_data(train,
                                      max_time=TIME_SEQ,
                                      max_demog=MAX_DEMOG,
                                      epochs=EPOCHS,
                                      multiclass=N_CLASS > 2,
                                      random_seed=RAND,
                                      resample=False,
                                      resample_frac=[0.9, 0.1],
                                      batch_size=BATCH_SIZE)

    validation_gen = tk.create_ragged_data(validation,
                                           max_time=TIME_SEQ,
                                           max_demog=MAX_DEMOG,
                                           epochs=EPOCHS,
                                           multiclass=N_CLASS > 2,
                                           random_seed=RAND,
                                           batch_size=BATCH_SIZE)

    # NOTE: don't shuffle test data
    test_gen = tk.create_ragged_data(test,
                                     max_time=TIME_SEQ,
                                     max_demog=MAX_DEMOG,
                                     epochs=1,
                                     multiclass=N_CLASS > 2,
                                     shuffle=False,
                                     random_seed=RAND,
                                     batch_size=BATCH_SIZE)

    # %% Setting up the model
    model = tk.LSTM(time_seq=TIME_SEQ,
                    vocab_size=N_VOCAB,
                    n_classes=N_CLASS,
                    n_demog=N_DEMOG,
                    n_demog_bags=MAX_DEMOG,
                    ragged=True,
                    lstm_dropout=LSTM_DROPOUT,
                    recurrent_dropout=LSTM_RECURRENT_DROPOUT,
                    batch_size=BATCH_SIZE)

    model.compile(optimizer="adam",
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=[
                      keras.metrics.AUC(num_thresholds=int(1e5),
                                        name="ROC-AUC"),
                      keras.metrics.AUC(num_thresholds=int(1e5),
                                        curve="PR",
                                        name="PR-AUC"),
                  ])

    model.summary()

    # %% Create Tensorboard callback
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
    fitting = model.fit(train_gen,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=validation_gen,
                        validation_steps=VALID_STEPS_PER_EPOCH,
                        epochs=EPOCHS,
                        callbacks=[
                            tb_callback, model_checkpoint_callback,
                            stopping_checkpoint
                        ])

    # Test
    print(model.evaluate(test_gen))

    # %% Validation F1 cut

    y_pred_validation = model.predict(validation_gen,
                                      steps=VALID_STEPS_PER_EPOCH)

    y_true_validation = [lab for _, _, lab in validation]

    # Resizing for output which is divisible by BATCH_SIZE
    y_true_validation = np.array(
        y_true_validation[0:y_pred_validation.shape[0]])

    val_gm = ta.grid_metrics(y_true_validation,
                             y_pred_validation,
                             min=0.0,
                             max=1.0,
                             step=0.001)

    f1_cut = val_gm.cutoff.values[np.argmax(val_gm.f1)]
    # %% Predicting on test data
    y_pred_test = model.predict(test_gen)

    y_true_test = [lab for _, _, lab in test]
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
