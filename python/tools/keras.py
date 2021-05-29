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


def sequence_to_multihot_tensor(y: list):

    max_feat = max([max(val) for val in y])

    y_mat = np.zeros((len(y), max_feat))
    for row, cols in enumerate(y):
        for col in cols:
            y_mat[row, col - 1] = 1

    return tf.convert_to_tensor(y_mat)

def sequence_to_onehot_tensor(y: list):
    max_feat = max([max(val) for val in y])

    feat_ragged = tf.ragged.constant(y)

    feat_one_hot = tf.one_hot(feat_ragged, max_feat)
    
    return feat_one_hot


def create_ragged_data_gen(inputs: list,
                           max_demog: int,
                           epochs: int,
                           multiclass: bool = False,
                           demog_output: str = None,
                           batch_size: int = 32,
                           random_seed: int = 1234,
                           ragged: bool = True,
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
    # if demog_output == "multi-hot":
    #     demog = sequence_to_multihot_tensor([dem for _, dem, _ in inputs])
    # elif demog_output == "one-hot":
    #     demog = sequence_to_onehot_tensor([dem for _, dem, _ in inputs])
    # else:
    # Take as sequence
    # BUG: Model doesn't seem to like this when it's ragged. Figure it out eventually.
    demog = tf.ragged.constant([dem for _, dem, _ in inputs])
    demog = demog.to_tensor(default_value=0,
                            shape=(demog.shape[0], max_demog))

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

# Hyperparameter Model builder
class LSTMHyper(kerastuner.HyperModel):
    """LSTM model with hyperparameter tuning.

    This is the first-draft LSTM model with a single embedding layer
    and LSTM layer.

    Input is assumed to be ragged on inner 2 dims.

    Args:
        vocab_size (int): Vocabulary size for embedding layer
        metrics: a keras metric or list of keras metrics to compile with
        loss: a keras loss function to minimize (optional)
        n_classes: Number of classes being predicted
        n_demog (int): Maximum number of demographic or non-time-varying features to be fed into
            the demog layer
        n_demog_bags (int): Maximum size of "bag" containing all demog feautures for a single sample
    """
    def __init__(self, vocab_size, metrics, loss = None, n_classes=1, n_demog=32, n_demog_bags=6):
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.n_demog = n_demog
        self.n_demog_bags = n_demog_bags
        self.metrics = metrics
        self.loss = loss

    def build(self, hp: kerastuner.HyperParameters) -> keras.Model:
        """Build LSTM model

        Notes:
            This is normally called within a HyperModel context.
        Args:
            hp (:obj:`HyperParameters`): `HyperParameters` instance

        Returns:
            A built/compiled keras model ready for hyperparameter tuning
        """

        # L1/L2 vals
        reg_vals = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

        # Model Topology

        # Should we multiply the feature embeddings by their averages?
        weighting = hp.Boolean("Feature Weighting")

        # Should we add a dense layer between RNN and output?
        final_dense = hp.Boolean("Final Dense Layer")

        # Feature Embedding Params
        emb_l1 = hp.Choice("Feature Embedding L1", reg_vals)
        emb_l2 = hp.Choice("Feature Embedding L2", reg_vals)

        emb_n = hp.Int("Embedding Dimension",
                                    min_value=64,
                                    max_value=512,
                                    default=64,
                                    step=64)

        # Demog Embedding
        demog_emb_n = hp.Int("Demographics Embedding Dimension",
                            min_value=1,
                            max_value=64,
                            default=self.n_demog
                            )

        # Average Embedding Params
        avg_l1 = hp.Choice("Average Embedding L1", reg_vals,
                                    parent_name = "Feature Weighting",
                                    parent_values = [True])
        avg_l2 = hp.Choice("Average Embedding L2", reg_vals,
                                    parent_name = "Feature Weighting",
                                    parent_values = [True])

        # LSTM Params
        lstm_n = hp.Int("LSTM Units",
                        min_value=32,
                        max_value=512,
                        default=32,
                        step=32)
        lstm_dropout = hp.Float("LSTM Dropout",
                                min_value=0.0,
                                max_value=0.9,
                                default=0.4,
                                step=0.01)
        lstm_recurrent_dropout = hp.Float("LSTM Recurrent Dropout",
                                            min_value=0.0,
                                            max_value=0.9,
                                            default=0.4,
                                            step=0.01)
        lstm_l1 = hp.Choice("LSTM weights L1", reg_vals)
        lstm_l2 = hp.Choice("LSTM weights L2", reg_vals)
        
        # Final dense layer
        dense_n = hp.Int("Dense Units",
                         min_value=2,
                         max_value=128,
                         sampling="log",
                         parent_name="Final Dense Layer",
                         parent_values=[True]
                         )
        # Model code
        feat_input = keras.Input(shape=(None, None), ragged=True)
        demog_input = keras.Input(shape=(self.n_demog_bags, ))

        demog_emb = keras.layers.Embedding(self.n_demog,
                                        output_dim=demog_emb_n,
                                        mask_zero=True,
                                       name="Demographic_Embeddings"
        )(demog_input)

        demog_avg = keras.layers.Flatten()(demog_emb)

        emb1 = keras.layers.Embedding(self.vocab_size,
                                    output_dim=emb_n,
                                    embeddings_regularizer=keras.regularizers.l1_l2(emb_l1, emb_l2),
                                    mask_zero=True,
                                    name="Feature_Embeddings")(feat_input)
        
        if weighting:
            emb2 = keras.layers.Embedding(self.vocab_size,
                                          output_dim=1,
                                          embeddings_regularizer=keras.regularizers.l1_l2(avg_l1, avg_l2),
                                          mask_zero=True,
                                          name="Average_Embeddings")(feat_input)

            # Multiplying the code embeddings by their respective weights
            mult = keras.layers.Multiply(name="Embeddings_by_Average")([emb1, emb2])
            avg = keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=2), name="Averaging")(mult)
        else:
            avg = keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=2), name="Averaging")(emb1)
        

        lstm_layer = keras.layers.LSTM(lstm_n, 
                               dropout=lstm_dropout,
                               recurrent_dropout=lstm_recurrent_dropout,
                               recurrent_regularizer=keras.regularizers.l1_l2(lstm_l1, lstm_l2),
                               name="Recurrent")(avg)
        
        lstm_layer = keras.layers.Concatenate()([lstm_layer, demog_avg])

        if final_dense:
            lstm_layer = keras.layers.Dense(dense_n, activation = "relu", name = "pre_output")(lstm_layer)

        activation_fn = "softmax" if self.n_classes > 2 else "sigmoid"
        output = keras.layers.Dense(
            self.n_classes if self.n_classes > 2 else 1,
            activation=activation_fn,
            name="Output")(lstm_layer)

        model = keras.Model([feat_input, demog_input], output)

        # --- Learning rate and momentum
        lr = hp.Choice("Learning Rate", [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
        momentum = hp.Float("Momentum", min_value=0.0, max_value=0.9, step=0.1)
        opt = keras.optimizers.SGD(lr, momentum=momentum)
        
        # --- Loss FN
        # NOTE: I was messing around with focal loss here, but I think that's
        # harder to justify and explain in this context
        if self.loss is None:
            if self.n_classes > 2:
            loss_fn = keras.losses.categorical_crossentropy
                else:
            loss_fn = keras.losses.binary_crossentropy
        
        model.compile(optimizer = opt, loss=self.loss, metrics=self.metrics)

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

    # # Embedding the demographic variables
    demog_emb = keras.layers.Embedding(n_demog,
                                       output_dim=lstm_dim,
                                       mask_zero=True,
                                       name="Demographic_Embeddings")(demog_in)

    # # Averaging the demographic variable embeddings
    demog_avg = keras.backend.mean(demog_emb, axis=2)

    # Concatenating the LSTM output and deemographic variable embeddings
    comb = keras.layers.Concatenate()([lstm_layer, demog_avg])

    # Running the embeddings through a final dense layer for prediction
    output = keras.layers.Dense(
        # BUG: We use a single output for the binary case but 3 for the multiclass case
        # so this should be able to account for that.
        n_classes if n_classes > 2 else 1,
        activation="softmax" if n_classes > 2 else "sigmoid",
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
    if n_classes > 1:
        activation = 'softmax'
    else:
        activation = 'sigmoid'
    output = keras.layers.Dense(n_classes,
                                activation=activation,
                                name='output')(dense)

    return keras.Model(input, output)

# Jacked from https://github.com/Tony607/Focal_Loss_Keras/blob/master/src/keras_focal_loss.ipynb
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2., alpha=4.,
                 reduction=keras.losses.Reduction.AUTO, name='focal_loss'):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})
        """
        super(FocalLoss, self).__init__(reduction=reduction,
                                        name=name)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        """
        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(
            tf.subtract(1., model_out), self.gamma))
        fl = tf.multiply(self.alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        
        return tf.reduce_mean(reduced_fl)