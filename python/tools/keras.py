'''Support classes and functions for Keras'''

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle as pkl

from tensorflow import keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import backend as K


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
