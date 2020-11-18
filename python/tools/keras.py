'''Support classes and functions for Keras'''

import numpy as np
import pandas as pd
import tensorflow as tf
import kerastuner as kt

import pickle as pkl

from tensorflow import keras as keras
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from kerastuner import HyperModel


class DataGenerator(keras.utils.Sequence):
    ''''Generates data for Keras. Code jacked from here:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    
    '''
    def __init__(self, 
                 inputs, 
                 labels,
                 dim, 
                 batch_size=32, 
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.inputs = inputs
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.inputs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate idx of the batch
        idx = self.idx[index*self.batch_size:(index+1)*self.batch_size]
            
        # Generate data
        X, y = self.__data_generation(idx)
        
        return X, y
    
    def on_epoch_end(self):
        'Updates idx after each epoch'
        self.idx = np.arange(len(self.inputs))
        if self.shuffle == True:
            np.random.shuffle(self.idx)
    
    def __data_generation(self, idx):
        'Generates data containing batch_size samples' 
        # Find list of IDs
        bags = [self.inputs[k] for k in idx]
        
        # Padding the feature bags
        padded_bags = [pad_sequences(bag, self.dim[-1]) for bag in bags]
        
        # Padding the visit sequences
        padded_seqs = pad_sequences(padded_bags, self.dim[0], value=[[0]])
        
        # Stacking the visits into a single array
        X = np.stack(padded_seqs).astype(np.uint32)
        
        return X, self.labels[idx]

class LSTMHyperModel(HyperModel):
    """LSTM model with hyperparameter tuning.

    This is the first-draft LSTM model with a single embedding layer
    and LSTM layer.

    Args:
        ragged (bool): Should the input be treated as ragged or dense?
        n_timesteps (int): length of time sequence
        n_tokens (int): Vocabulary size for embedding layer
        batch_size (int): Training batch size
    """
    def __init__(self, ragged, n_timesteps, n_tokens, batch_size):
        # Capture model parameters at init
        self.ragged = ragged
        self.n_timesteps = n_timesteps
        self.n_tokens = n_tokens
        self.batch_size = batch_size

    def build(self, hp):
        """Build LSTM model 

        Notes:
            This is normally called within a HyperModel context.
        Args:
            hp (:obj:`HyperParameters`): `HyperParameters` instance

        Returns:
            A built/compiled keras model ready for hyperparameter tuning
        """
        # NOTE: Not doing sequential here incase we want
        # to use multiple inputs / multi embedding
        inp = Input(
            shape=(n_timesteps, None if self.ragged else n_tokens),
            ragged = self.ragged,
            batch_size=self.batch_size
        )
        emb = Embedding(
            input_dim=self.n_tokens
        )(inp)
        lstm = LSTM(
            units=hp.Int("LSTM Units", min_value=32, max_value=512, step=32),
            dropout=hp.Float("LSTM Dropout", min_value=0., max_value=0.9, step=0.05),
            recurrent_dropout=hp.Float("LSTM Recurrent Dropout", min_value=0., max_value=0.9, step=0.05)
        )(emb)
        output = Dense(1, activation="relu")(lstm)

        model = keras.Model(inp, output)

        model.compile(
            optimizer=keras.optimizers.Adagrad(
                # NOTE: we could also use a LR adjustment callback on train
                # instead.
                hp.Choice("Learning Rate", values=[1e-2, 1e-3, 1e-4])
            ),
            # NOTE: Assuming binary classification task, but we could change.
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        return model