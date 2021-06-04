"""
Starter Keras model
"""
# %%
import csv
import os
import pickle as pkl
import pandas as pd
import argparse

import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import TensorBoard

import tools.analysis as ta
from tools import keras as tk
import tools.preprocessing as tp

# %% Globals
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model",
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
                        default="output/",
                        help="output directory")
    parser.add_argument("--data_dir",
                        type=str,
                        default="../data/data/",
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
    MOD_NAME = args.model
    TARGET = args.outcome
    DEMOG = args.demog
    DAY_ONE_ONLY = args.day_one
    LSTM_DROPOUT = args.dropout
    LSTM_RECURRENT_DROPOUT = args.recurrent_dropout
    N_LSTM = args.n_cells
    BATCH_SIZE = args.batch_size
    WEIGHTED_LOSS = args.weighted_loss
    EPOCHS = args.epochs
    TEST_SPLIT = args.test_split
    VAL_SPLIT = args.validation_split
    RAND = args.rand_seed
    TB_UPDATE_FREQ = args.tb_update_freq

    # DIRS / FILES
    output_dir = os.path.abspath(args.out_dir)
    tensorboard_dir = os.path.abspath(
        os.path.join(args.data_dir, "..", "model_checkpoints"))
    data_dir = os.path.abspath(args.data_dir)
    pkl_dir = os.path.join(output_dir, "pkl")
    stats_dir = os.path.join(output_dir, "analysis")

    # Create analysis dir if it doesn't exist
    os.makedirs(stats_dir, exist_ok=True)

    stats_filename = TARGET + "_stats.csv"

    # Data load
    with open(os.path.join(pkl_dir, TARGET + "_trimmed_seqs.pkl"), "rb") as f:
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
    N_CLASS = max(x for _, _, x in inputs) + 1
    
    # Setting y here so it's stable
    y = np.array([l[2] for l in inputs])

    # Create some callbacks
    callbacks = [
        # TensorBoard(
        #     log_dir=os.path.join(tensorboard_dir, TARGET,
        #                          datetime.now().strftime("%Y%m%d-%H%M%S")),
        #     histogram_freq=1,
        #     update_freq=TB_UPDATE_FREQ,
        #     embeddings_freq=5,
        #     embeddings_metadata=os.path.join(tensorboard_dir,
        #                                      "emb_metadata.tsv"),
        # ),

        # Create model checkpoint callback
        # keras.callbacks.ModelCheckpoint(filepath=os.path.join(
        #     tensorboard_dir, TARGET,
        #     "weights.{epoch:02d}-{val_loss:.2f}.hdf5"),
        #                                 save_weights_only=True,
        #                                 monitor="val_loss",
        #                                 mode="max",
        #                                 save_best_only=True),

        # Create early stopping callback
        keras.callbacks.EarlyStopping(monitor="val_loss",
                                      min_delta=0,
                                      patience=2,
                                      mode="auto")
    ]

    # Create some metrics
    metrics = [
        keras.metrics.AUC(num_thresholds=int(1e5), name="ROC-AUC"),
        keras.metrics.AUC(num_thresholds=int(1e5), curve="PR", name="PR-AUC"),
    ]

    # Define loss function
    # NOTE: We were experimenting with focal loss at one point, maybe we can try that again at some point
    if TARGET ==  'multi_class':
        loss_fn = keras.losses.categorical_crossentropy
    else:
        loss_fn = keras.losses.binary_crossentropy
    
    # Splitting the data
    train, test = train_test_split(
        range(len(inputs)),
        test_size=TEST_SPLIT,
        stratify=y,
        random_state=RAND)
    
    train, val = train_test_split(
        train,
        test_size=VAL_SPLIT,
        stratify=y[train],
        random_state=RAND)
    
    # Optional weighting
    if WEIGHTED_LOSS:
        MOD_NAME += '_w'
        classes = np.unique(y)
        weights = compute_class_weight('balanced',
                                       classes=classes,
                                       y=y[train])
        weight_dict = {c: weights[i] for i, c in enumerate(classes)}
    else:
        weight_dict = None
    
    # --- Model-specific code
    # TODO: Rework some sections to streamline input and label generation along with splitting
    if "lstm" in MOD_NAME:
        # %% Compute steps-per-epoch
        # NOTE: Sometimes it can't determine this properly from tf.data
        STEPS_PER_EPOCH = np.ceil(len(train) / BATCH_SIZE)
        VALID_STEPS_PER_EPOCH = np.ceil(len(val) / BATCH_SIZE)

        # %%
        train_gen = tk.create_ragged_data_gen([inputs[samp] for samp in train],
                                              max_demog=MAX_DEMOG,
                                              epochs=1,
                                              multiclass=N_CLASS > 2,
                                              random_seed=RAND,
                                              batch_size=BATCH_SIZE)

        validation_gen = tk.create_ragged_data_gen(
            [inputs[samp] for samp in val],
            max_demog=MAX_DEMOG,
            epochs=1,
            shuffle=False,
            multiclass=N_CLASS > 2,
            random_seed=RAND,
            batch_size=BATCH_SIZE)

        # NOTE: don't shuffle test data
        test_gen = tk.create_ragged_data_gen([inputs[samp] for samp in test],
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
                        recurrent_dropout=LSTM_RECURRENT_DROPOUT)

        model.compile(optimizer="adam", loss=loss_fn, metrics=metrics)

        # Train
        fitting = model.fit(train_gen,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_data=validation_gen,
                            validation_steps=VALID_STEPS_PER_EPOCH,
                            epochs=EPOCHS,
                            callbacks=callbacks,
                            class_weight=weight_dict)
        
    elif "dan" in MOD_NAME:
        if DAY_ONE_ONLY:
            # Optionally limiting the features to only those from the first day
            # of the actual COVID visit
            MOD_NAME += "_d1"
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
                                                       maxlen=225,
                                                       padding='post')
        
        # Handle multiclass case
        if N_CLASS > 2:
            # We have to pass one-hot labels for model fit, but CLF metrics 
            # will take indices
            n_values = np.max(y) + 1
            y_one_hot = np.eye(n_values)[y]

            # Produce DAN model to fit
            model = tk.DAN(
                vocab_size=N_VOCAB,
                # TODO: Maybe parameterize? Ionno.
                ragged=False,
                input_length=TIME_SEQ,
                n_classes=N_CLASS)

            model.compile(optimizer="adam", loss=loss_fn, metrics=metrics)

            model.fit(X[train],
                      y_one_hot[train],
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      validation_data=(X[val], y_one_hot[val]),
                      callbacks=callbacks)
        
        else:
            # Produce DAN model to fit
            model = tk.DAN(
                vocab_size=N_VOCAB,
                # TODO: Maybe parameterize? Ionno.
                ragged=False,
                input_length=TIME_SEQ)

            model.compile(optimizer="adam", loss=loss_fn, metrics=metrics)

            model.fit(X[train],
                      y[train],
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      validation_data=(X[val], y[val]),
                      callbacks=callbacks)            

    # ---
    # Compute decision threshold cut from validation data using grid search
    # then apply threshold to test data to compute metrics
    val_probs = model.predict(validation_gen)
    test_probs = model.predict(test_gen)
    
    # model.save(output_dir + '/' + MOD_NAME)
    
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
        
        # Writing the predicted probabilities to disk
        probs_file = '/probs/' + MOD_NAME + '_' + TARGET + '.pkl'
        prob_out = {'cutpoint': cutpoint, 'probs': test_probs}
        pkl.dump(prob_out, open(stats_dir + probs_file, 'wb'))
        
        # Getting the test predictions
        test_preds = ta.threshold(test_probs, cutpoint)
        
    else:
        # Getting the stats
        stats = ta.clf_metrics(y[test],
                               test_probs,
                               average="weighted",
                               mod_name=MOD_NAME)
        
        # Writing the predicted probabilities to disk
        probs_file = '/probs/' + MOD_NAME + '_' + TARGET + '.pkl'
        prob_out = {'cutpoint': 0.5, 'probs': test_probs}
        pkl.dump(prob_out, open(stats_dir + probs_file, 'wb'))
        
        # Getting the test predictions
        test_preds = np.argmax(test_probs, axis=1)
        
        # Saving the max from each row for writing to CSV
        test_probs = np.amax(test_probs, axis=1)
    
    # ---
    # Writing the results to disk
    # Optionally append results if file already exists
    append_file = stats_filename in os.listdir(stats_dir)
    preds_filename = TARGET + "_preds.csv"
    
    stats.to_csv(os.path.join(stats_dir, stats_filename),
                 mode="a" if append_file else "w",
                 header=False if append_file else True,
                 index=False)

    # Writing the test predictions to the test predictions CSV
    if preds_filename in os.listdir(stats_dir):
        preds_df = pd.read_csv(os.path.join(stats_dir, preds_filename))
    else:
        preds_df = pd.read_csv(os.path.join(output_dir,
                                            TARGET + "_cohort.csv"))
        preds_df = preds_df.iloc[test, :]

    preds_df[MOD_NAME + '_prob'] = test_probs
    preds_df[MOD_NAME + '_pred'] = test_preds
    preds_df.to_csv(os.path.join(stats_dir, preds_filename), index=False)