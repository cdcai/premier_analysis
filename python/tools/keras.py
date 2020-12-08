"""Support classes and functions for Keras"""

import itertools
import pickle as pkl

import kerastuner
import numpy as np
import pandas as pd
import tensorflow as tf
from kerastuner import HyperModel
from sklearn.utils import _safe_indexing
from tensorflow import keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (LSTM, Dense, Embedding, Input, Multiply,
                                     Reshape)
from tensorflow.keras.preprocessing.sequence import pad_sequences


def create_ragged_data(inputs: tuple,
                       max_time: float,
                       epochs: int,
                       batch_size: int = 32,
                       random_seed: int = 1234,
                       ragged: bool = True,
                       resample: bool = False,
                       resample_frac=[0.9, 0.1],
                       label_int: bool = False,
                       shuffle: bool = True) -> tf.data.Dataset:
    """A tf.dataset generator which handles both ragged and dense data accordingly"""
    # Check that empty lists are converted to zeros
    x = [[(lambda x: [0] if x == [] else x)(bags) for bags in seq]
         for seq, _ in inputs]

    # Convert to ragged
    # shape: (len(x), None, None)
    X = tf.ragged.constant(x)

    # Sanity check
    assert X.shape.as_list() == [len(x), None, None]

    if not ragged:
        X.to_tensor()

    # Labs as stacked
    # NOTE: some loss functions require this to be float
    y = np.array([tup[1] for tup in inputs],
                 dtype=np.int32 if label_int else np.float)

    # Make sure our data are equal
    assert y.shape[0] == X.shape[0]

    # Produce data generator
    if resample:
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        pos_data = tf.data.Dataset.from_tensor_slices(
            (tf.gather(X, pos_idx), y[pos_idx]))
        neg_data = tf.data.Dataset.from_tensor_slices(
            (tf.gather(X, neg_idx), y[neg_idx]))

        data_gen = tf.data.experimental.sample_from_datasets(
            datasets=[neg_data, pos_data],
            weights=resample_frac,
            seed=random_seed)

    else:
        data_gen = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        data_gen = data_gen.shuffle(buffer_size=len(x),
                                    seed=random_seed,
                                    reshuffle_each_iteration=True)

    data_gen = data_gen.repeat(epochs)

    data_gen = data_gen.batch(batch_size)

    return data_gen


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras.

    Code jacked from here:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(
        self,
        inputs,
        max_time=225,
        ragged=False,
        labels=None,
        resampler=None,
        batch_size=32,
        shuffle=True,
    ):
        """Initializes the data generator.

        Args:
          inputs: either a list of lists of lists of integers, or a list
            of visit-level tuples of the form ([list of lists of
            integers], label)
          max_time (int): The target time step length for each visit after padding
          ragged (bool): Should the input data be treated as a RaggedTensor, or made dense by batch?
          labels: a np.array of sequence-level labels
          resampler: An imblearn sampler object if resampling is desired, or None (default: None)
          batch_size: size for the minibatches
          shuffle: whether to shuffle the data after each epoch

        Note:
            If using resampler, the sampler must provide indicies ()
        """
        # Setting some basic attributes
        self.max_time = max_time
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = resampler
        self.ragged = ragged
        # Separating the inputs and labels, in case they were passed as one
        if labels is None:
            labels = np.array([tup[1] for tup in inputs])
            inputs = [tup[0] for tup in inputs]

        # NOTE: Assert that the user is not a clown
        assert len(inputs) == len(
            labels), "Inputs and labels are not of equal length"

        self.inputs = inputs
        self.labels = labels

        # If resampled, fit resampler and assign idx
        # Shuffling the data to start
        # FIXME
        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches per epoch"""
        n_batch = int(np.floor(len(self.inputs) / self.batch_size))
        return n_batch
    
    def __getitem__(self, index):
        """Fetches a batch of X, y pairs given a batch number"""
        # Generate idx of the batch
        idx = self.indices_[index * self.batch_size:(index + 1) *
                            self.batch_size]

        # Generate data for the batch
        X, y = self.__data_generation(idx)

        return X, y

    def on_epoch_end(self):
        """Decides what to do at the end of each epoch, e.g., shuffling"""
        if self.sampler is None:
            self.indices_ = np.arange(len(self.inputs))
        else:
            # NOTE: Shamelessly jacked from imblearn
            self.sampler.fit_resample(
                np.array(self.inputs).reshape(-1, 1), self.labels)
            if not hasattr(self.sampler, "sample_indices_"):
                raise ValueError("'sampler' needs to have an attribute "
                                 "'sample_indices_'.")

            self.indices_ = self.sampler.sample_indices_
        if self.shuffle == True:
            np.random.shuffle(self.indices_)

    def __data_generation(self, idx):
        """Yields a batch of X, y pairs given batch indices"""
        # Making a list of the visit sequences in the batch
        X = _safe_indexing(self.inputs, idx)

        # Fix issue where empty lists are propagated
        X = [[(lambda x: [0] if x == [] else x)(bags) for bags in seq]
             for seq in X]
        if self.ragged:
            # Pad to max time
            x_pad = [l + [[0]] * (self.max_time - len(l)) for l in X]

            # Convert to ragged
            # shape: (self.batch_size, self.max_time, None)
            x_rag = tf.RaggedTensor.from_uniform_row_length(
                tf.ragged.constant(list(itertools.chain.from_iterable(x_pad))),
                self.max_time)

            # Sanity check
            assert x_rag.shape.as_list() == [
                self.batch_size, self.max_time, None
            ]

            X = x_rag
        else:
            # Figuring out how much to pad the bags
            try:
                biggest_bag = np.max([[len(bag) for bag in seq]
                                      for seq in X])[0]
            except IndexError:
                # FIXME: No idea why this errors but here's a bodge
                biggest_bag = np.max([[len(bag) for bag in seq] for seq in X])

            # Padding the feature bags in each visit to V
            padded_bags = [
                pad_sequences(seq, biggest_bag, padding="post") for seq in X
            ]

            # Padding each visit sequence to MAX_TIME
            padded_seqs = pad_sequences(padded_bags,
                                        self.max_time,
                                        value=[[0]])

            # Stacking the fully-padded sequences of bags into a single array and
            # selecting the corresponding labels
            X = np.stack(padded_seqs).astype(np.uint32)

        y = _safe_indexing(self.labels, idx)

        return X, y


class LSTMHyperModel(HyperModel):
    """LSTM model with hyperparameter tuning.

    This is the first-draft LSTM model with a single embedding layer
    and LSTM layer.

    Args:
        ragged (bool): Should the input be treated as ragged or dense?
        n_timesteps (int): length of time sequence
        vocab_size (int): Vocabulary size for embedding layer
        batch_size (int): Training batch size
    """
    def __init__(self, ragged: bool, n_timesteps: int, vocab_size: int,
                 batch_size: int):
        # Capture model parameters at init
        self.ragged = ragged
        self.n_timesteps = n_timesteps
        self.vocab_size = vocab_size
        self.batch_size = batch_size

    def build(self, hp: kerastuner.HyperParameters) -> keras.Model:
        """Build LSTM model

        Notes:
            This is normally called within a HyperModel context.
        Args:
            hp (:obj:`HyperParameters`): `HyperParameters` instance

        Returns:
            A built/compiled keras model ready for hyperparameter tuning
        """

        inp = Input(
            shape=(self.n_timesteps, None),
            ragged=self.ragged,
            batch_size=self.batch_size,
            name="Input",
        )
        emb1 = Embedding(
            input_dim=self.vocab_size,
            mask_zero=True,
            embeddings_regularizer=keras.regularizers.l1_l2(
                l1=hp.Float("Feature Embedding L1",
                            min_value=0.0,
                            max_value=0.2,
                            step=0.05),
                l2=hp.Float("Feature Embedding L2",
                            min_value=0.0,
                            max_value=0.2,
                            step=0.05)),
            output_dim=hp.Int("Embedding Dimension",
                              min_value=64,
                              max_value=1024,
                              step=64),
            name="Feature_Embeddings",
        )(inp)
        emb2 = Embedding(input_dim=self.vocab_size,
                         output_dim=1,
                         mask_zero=True,
                         embeddings_regularizer=keras.regularizers.l1_l2(
                             l1=hp.Float("Average Embedding L1",
                                         min_value=0.0,
                                         max_value=0.2,
                                         step=0.05),
                             l2=hp.Float("Average Embedding L2",
                                         min_value=0.0,
                                         max_value=0.2,
                                         step=0.05)),
                         name="Average_Embeddings")(inp)
        mult = Multiply(name="Embeddings_by_Average")([emb1, emb2])
        avg = K.mean(mult, axis=2)
        lstm = LSTM(
            units=hp.Int("LSTM Units", min_value=32, max_value=512, step=32),
            dropout=hp.Float("LSTM Dropout",
                             min_value=0.0,
                             max_value=0.9,
                             step=0.05),
            recurrent_dropout=hp.Float("LSTM Recurrent Dropout",
                                       min_value=0.0,
                                       max_value=0.9,
                                       step=0.05),
            activity_regularizer=keras.regularizers.l1_l2(
                l1=hp.Float("LSTM Activation L1",
                            min_value=0.0,
                            max_value=0.2,
                            step=0.05),
                l2=hp.Float("LSTM Activation L2",
                            min_value=0.0,
                            max_value=0.2,
                            step=0.05)),
            kernel_regularizer=keras.regularizers.l1_l2(
                l1=hp.Float("LSTM weights L1",
                            min_value=0.0,
                            max_value=0.2,
                            step=0.05),
                l2=hp.Float("LSTM weights L2",
                            min_value=0.0,
                            max_value=0.2,
                            step=0.05)),
            name="Recurrent",
        )(avg)
        output = Dense(1, activation="sigmoid", name="Output")(lstm)

        model = keras.Model(inp, output, name="LSTM-Hyper")

        model.compile(
            optimizer=keras.optimizers.SGD(
                            learning_rate=hp.Choice("Learning Rate",
                                    values=[1e-2, 1e-3, 1e-4])
            ),
            #keras.optimizers.Adam(),
            # NOTE: we could also use a LR adjustment callback on train
            # instead. We could also use any optimizer here

            # NOTE: Assuming binary classification task, but we could change.
            loss="binary_crossentropy",
            metrics=["accuracy"])

        return model
