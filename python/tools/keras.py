"""Support classes and functions for Keras"""

import numpy as np
import pandas as pd
import itertools
import pickle as pkl
import kerastuner
import tensorflow as tf
import tensorflow_addons as tfa

from kerastuner import HyperModel
from tensorflow import keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Dense, Embedding, Input, Multiply,
                                     Reshape)
from tensorflow.keras.preprocessing.sequence import pad_sequences


def create_ragged_data(inputs: list,
                       max_time: float,
                       max_demog: int,
                       epochs: int,
                       multiclass: bool = False,
                       batch_size: int = 32,
                       random_seed: int = 1234,
                       ragged: bool = True,
                       pad_shape=None,
                       resample: bool = False,
                       resample_frac=[0.9, 0.1],
                       label_int: bool = False,
                       shuffle: bool = True) -> tf.data.Dataset:
    """A tf.dataset generator which handles both ragged and dense data accordingly"""
    # Check that empty lists are converted to zeros
    seq = [[(lambda x: [0] if x == [] else x)(bags) for bags in seq]
           for seq, _, _ in inputs]

    # Convert to ragged
    # shape: (len(x), None, None)
    X = tf.ragged.constant(seq)

    # Sanity check
    assert X.shape.as_list() == [len(seq), None, None]

    # Making demographics dense
    # BUG: Model doesn't seem to like this when it's ragged. Figure it out eventually.
    demog = tf.ragged.constant([dem for _, dem, _ in inputs])

    demog = demog.to_tensor(default_value=0, shape=(demog.shape[0], max_demog))
    if not ragged:
        # This will be an expensive operation
        # and will probably not work.
        X = X.to_tensor()

    # Labs as stacked
    # NOTE: some loss functions require this to be float
    if multiclass:
        y = tf.one_hot([tup[2] for tup in inputs],
                       max([tup[2] for tup in inputs]) + 1)
    else:
        y = np.array([tup[2] for tup in inputs],
                     dtype=np.int32 if label_int else np.float)

    # Make sure our data are equal
    assert y.shape[0] == X.shape[0]

    # Produce data generator
    if resample:
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        pos_samples = tf.data.Dataset.from_tensor_slices(
            (tf.gather(X, pos_idx), tf.gather(demog, pos_idx)))
        pos_labels = tf.data.Dataset.from_tensor_slices(y[pos_idx])

        pos_data = tf.data.Dataset.zip((pos_samples, pos_labels))

        neg_samples = tf.data.Dataset.from_tensor_slices(
            (tf.gather(X, neg_idx), tf.gather(demog, neg_idx)))
        neg_labels = tf.data.Dataset.from_tensor_slices(y[neg_idx])

        neg_data = tf.data.Dataset.zip((neg_samples, neg_labels))

        data_gen = tf.data.experimental.sample_from_datasets(
            datasets=[neg_data, pos_data],
            weights=resample_frac,
            seed=random_seed)

    else:
        data_samp = tf.data.Dataset.from_tensor_slices((X, demog))
        data_lab = tf.data.Dataset.from_tensor_slices(y)

        data_gen = tf.data.Dataset.zip((data_samp, data_lab))
    if shuffle:
        data_gen = data_gen.shuffle(buffer_size=len(seq),
                                    seed=random_seed,
                                    reshuffle_each_iteration=True)

    data_gen = data_gen.batch(batch_size)

    data_gen = data_gen.repeat(epochs)

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
        bias_init (float): Starting bias of the output layer (optional)
    """
    def __init__(self,
                 ragged: bool,
                 n_timesteps: int,
                 vocab_size: int,
                 batch_size: int,
                 bias_init: float = None):
        # Capture model parameters at init
        self.ragged = ragged
        self.n_timesteps = n_timesteps
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.bias_init = bias_init if bias_init is not None else 0.0

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
            shape=(None if self.ragged else self.n_timesteps, None),
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
                            max_value=0.1,
                            step=0.01),
                l2=hp.Float("Feature Embedding L2",
                            min_value=0.0,
                            max_value=0.1,
                            step=0.01)),
            output_dim=hp.Int("Embedding Dimension",
                              min_value=64,
                              max_value=512,
                              default=64,
                              step=64),
            name="Feature_Embeddings",
        )(inp)
        emb2 = Embedding(input_dim=self.vocab_size,
                         output_dim=1,
                         mask_zero=True,
                         embeddings_regularizer=keras.regularizers.l1_l2(
                             l1=hp.Float("Average Embedding L1",
                                         min_value=0.0,
                                         max_value=0.1,
                                         step=0.01),
                             l2=hp.Float("Average Embedding L2",
                                         min_value=0.0,
                                         max_value=0.1,
                                         step=0.01)),
                         name="Average_Embeddings")(inp)
        if self.ragged:
            mult = keras.layers.Multiply(name="Embeddings_by_Average")(
                [emb1, emb2])
            avg = keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=2),
                                      name="Averaging")(mult)
        else:
            mult = Multiply(name="Embeddings_by_Average")([emb1, emb2])
            avg = K.mean(mult, axis=2)

        lstm = keras.layers.LSTM(units=hp.Int("LSTM Units",
                                              min_value=32,
                                              max_value=512,
                                              default=32,
                                              step=32),
                                 dropout=hp.Float("LSTM Dropout",
                                                  min_value=0.0,
                                                  max_value=0.9,
                                                  default=0.4,
                                                  step=0.01),
                                 recurrent_dropout=hp.Float(
                                     "LSTM Recurrent Dropout",
                                     min_value=0.0,
                                     max_value=0.9,
                                     default=0.4,
                                     step=0.01),
                                 activity_regularizer=keras.regularizers.l1_l2(
                                     l1=hp.Float("LSTM Activation L1",
                                                 min_value=0.0,
                                                 max_value=0.1,
                                                 step=0.01),
                                     l2=hp.Float("LSTM Activation L2",
                                                 min_value=0.0,
                                                 max_value=0.1,
                                                 step=0.01)),
                                 kernel_regularizer=keras.regularizers.l1_l2(
                                     l1=hp.Float("LSTM weights L1",
                                                 min_value=0.0,
                                                 max_value=0.1,
                                                 step=0.01),
                                     l2=hp.Float("LSTM weights L2",
                                                 min_value=0.0,
                                                 max_value=0.1,
                                                 step=0.01)),
                                 name="Recurrent")(avg)
        output = Dense(1,
                       activation="sigmoid",
                       name="Output",
                       bias_initializer=tf.keras.initializers.Constant(
                           self.bias_init.item()))(lstm)

        model = keras.Model(inp, output, name="LSTM-Hyper")

        lr = hp.Choice("Learning Rate", [1e-2, 1e-3, 1e-4])
        momentum = hp.Choice("Momentum", [0.0, 0.2, 0.4, 0.6, 0.8, 0.9])

        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=lr,
                                           momentum=momentum),
            # NOTE: TFA version won't run in kerastuner for some reason
            # loss=tfa.losses.SigmoidFocalCrossEntropy()
            #     alpha=hp.Float("Balancing Factor",
            #                    min_value=0.25,
            #                    max_value=0.74,
            #                    step=0.25),
            #     gamma=hp.Float("Modulating Factor",
            #                    min_value=0.0,
            #                    max_value=5.0,
            #                    step=0.5,
            #                    default=2.0)),
            # NOTE: For gamma = 0 & alpha = 1, Focal loss = binary_crossentropy
            loss=BinaryFocalLoss(gamma=hp.Float("Modulating Factor",
                                                min_value=0.0,
                                                max_value=5.0,
                                                step=1.0,
                                                default=2.0),
                                 pos_weight=hp.Float("Balancing Factor",
                                                     min_value=0.0,
                                                     max_value=1.0,
                                                     default=0.25,
                                                     step=0.25)),
            metrics=[
                keras.metrics.AUC(num_thresholds=int(1e4), name="ROC-AUC"),
                keras.metrics.AUC(num_thresholds=int(1e4),
                                  curve="PR",
                                  name="PR-AUC")
            ])

        return model


def LSTM(time_seq,
         vocab_size,
         emb_dim=64,
         lstm_dim=32,
         lstm_dropout=0.2,
         recurrent_dropout=0.2,
         n_classes=1,
         n_demog_bags=6,
         n_demog=32,
         output_bias=0.0,
         weighted_average=True,
         ragged=True):
    # Input layer
    code_in = keras.Input(shape=(None if ragged else time_seq, None),
                          ragged=ragged)

    # Feature Embeddings
    emb1 = keras.layers.Embedding(vocab_size,
                                  output_dim=emb_dim,
                                  mask_zero=True,
                                  name="Feature_Embeddings")(code_in)

    # Optionally learning averaging weights for the embeddings
    if weighted_average:
        # Looking up the averaging weights for each code
        emb2 = keras.layers.Embedding(vocab_size,
                                      output_dim=1,
                                      mask_zero=True,
                                      name="Average_Embeddings")(code_in)

        # Multiplying the code embeddings by their respective weights
        mult = keras.layers.Multiply(name="Embeddings_by_Average")(
            [emb1, emb2])

        # Computing the mean of the weighted embeddings
        if ragged:
            # NOTE: I think these are the equivalent ragged-aware ops
            # but that could be incorrect
            avg = keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=2),
                                      name="Averaging")(mult)
        else:
            avg = keras.backend.mean(mult, axis=2)

    else:
        avg = keras.backend.mean(emb1, axis=2)

    # Running the sequences through the LSTM
    lstm_layer = keras.layers.LSTM(lstm_dim,
                                   dropout=lstm_dropout,
                                   recurrent_dropout=recurrent_dropout,
                                   name="Recurrent")(avg)

    # Bringing in the demographic variables
    demog_in = keras.Input(shape=(n_demog_bags, ))

    # Embedding the demographic variables
    demog_emb = keras.layers.Embedding(n_demog,
                                       output_dim=lstm_dim,
                                       mask_zero=True,
                                       name="Demographic_Embeddings")(demog_in)

    # Averaging the demographic variable embeddings
    demog_avg = keras.backend.mean(demog_emb, axis=2)

    # Concatenating the LSTM output and deemographic variable embeddings
    comb = keras.layers.Concatenate()([lstm_layer, demog_avg])

    # Running the embeddings through a final dense layer for prediction
    output = keras.layers.Dense(
        # BUG: We use a single output for the binary case but 3 for the multiclass case
        # so this should be able to account for that.
        n_classes if n_classes > 2 else 1,
        activation="sigmoid",
        bias_initializer=keras.initializers.Constant(output_bias),
        name="Output")(comb)

    return keras.Model([code_in, demog_in], output)


def DAN(vocab_size,
        ragged=True,
        input_length=None,
        embedding_size=64,
        dense_size=32,
        n_classes=1):
    '''A deep averaging network (DAN) with only a single dense layer'''
    # Specifying the input
    input = keras.Input(shape=(None if ragged else input_length, ),
                        ragged=ragged)

    # Feature Embeddings
    embeddings = keras.layers.Embedding(vocab_size,
                                        output_dim=embedding_size,
                                        mask_zero=True,
                                        name='embeddings')(input)

    # Averaging the embeddings
    embedding_avg = keras.backend.mean(embeddings, 1)

    # Dense layers
    dense = keras.layers.Dense(dense_size, name='dense_1')(embedding_avg)
    output = keras.layers.Dense(n_classes, activation='sigmoid',
                                name='output')(dense)

    return keras.Model(input, output)
