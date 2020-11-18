'''Support classes and functions for Keras'''

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle as pkl

from tensorflow import keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import backend as K


class DataGenerator(keras.utils.Sequence):
    '''Generates data for Keras.
    
    Code jacked from here:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    '''
    def __init__(self, 
                 inputs, 
                 dim,
                 labels=None, 
                 batch_size=32, 
                 shuffle=True):
        '''Initializes the data generator. 
        
        Args:
          inputs: either a list of lists of lists of integers, or a list
            of visit-level tuples of the form ([list of lists of 
            integers], label)
          dim: a tuple of the form (max_time, vocab_size) that specifies
            the goal (i.e., padded) size for a single input example
          labels: a np.array of sequence-level labels
          batch_size: size for the minibatches
          shuffle: whether to shuffle the data after each epoch
        '''
        # Setting some basic attributes
        self.dim = max_time
        self.batch_size = batch_size
        self.inputs = inputs
        self.shuffle = shuffle
        
        # Separating the inputs and labels, in case they were passed as one
        if labels is None:
            labels = np.array([tup[1] for tup in inputs])
            inputs = [tup[0] for tup in inputs]
        
        # Shuffling the data to start
        self.on_epoch_end()
    
    def __len__(self):
        '''Returns the number of batches per epoch'''
        n_batch = int(np.floor(len(self.inputs) / self.batch_size))
        return n_batch
    
    def __getitem__(self, index):
        '''Fetches a batch of X, y pairs given a batch number'''
        # Generate idx of the batch
        idx = self.idx[index*self.batch_size:(index+1)*self.batch_size]
            
        # Generate data for the batch
        X, y = self.__data_generation(idx)
        
        return X, y
    
    def on_epoch_end(self):
        '''Decides what to do at the end of each epoch, e.g., shuffling'''
        # Shuffles the data after each epoch
        self.idx = np.arange(len(self.inputs))
        if self.shuffle == True:
            np.random.shuffle(self.idx)
    
    def __data_generation(self, idx):
        '''Yields a batch of X, y pairs given batch indicies'''
        # Making a list of the visit sequences in the batch
        bags = [self.inputs[k] for k in idx]
        
        # Padding the feature bags in each visit to V
        padded_bags = [pad_sequences(bag, self.dim[-1]) for bag in bags]
        
        # Padding each visit sequence to MAX_TIME
        padded_seqs = pad_sequences(padded_bags, self.dim[0], value=[[0]])
        
        # Stacking the fully-padded sequences of bags into a single array and
        # selecting the corresponding labels
        X = np.stack(padded_seqs).astype(np.uint32)
        y = self.labels[idx]
        
        return X, y
