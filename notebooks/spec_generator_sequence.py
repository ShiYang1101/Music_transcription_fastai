import tensorflow as tf
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import OneHotEncoder
import math
import os
import glob
from spectrogram_class import spectrogram
import pandas as pd
import numpy as np

'''
Modules for generating dataset for tensorflow model.
Utilized tensorlow Sequence as parent class
'''

# Get path to notebook
dir_path = os.path.dirname(__file__)
# Get path to data 
_path_to_npy = glob.glob('../data/**[!MACOSC]/*OrchideaSOL2020/', recursive=True)[0]

# Get metadata df
meta_df = pd.read_csv('../data/OrchideaSOL_metadata.csv')

# We will be using one hot encoder for our instrument class
_onehot = OneHotEncoder(sparse=False)
_onehot.fit(meta_df[['Instrument (in full)']])

def _get_spec(path, test_verbose = False, live_generation = False, 
                preprocess = True, n_mels = 512):
    '''
    Support function for spec_generator_sequence class for generating
    spectrogram from path
    
    '''
    path = os.path.join(dir_path, _path_to_npy, path)
    if live_generation:
        spec = spectrogram(path, n_mels = n_mels, preprocess= preprocess)
        return spec.spec
    try:
        if test_verbose:
            print('HIT')
        return np.load(os.path.splitext(path)[0] + \
                                '.npy', allow_pickle=True)
        if test_verbose:
            print("SUCCESS")
    except:
        if test_verbose:
            print("FAILED")
        spec = spectrogram(path, n_mels = 512, preprocess= preprocess)
    # if preprocess = True:
    #     spec.
    # spec.spec = np.expand_dims(spec.spec, -1)
    return spec.spec

class spec_generator(Sequence):

    def __init__(self, df, batch_size, add_channel = False, live_generation = False, 
                    preprocess = True, n_mels = 512):
        self.x = df['Path'].array
        self.y = _onehot.transform(df[['Instrument (in full)']])
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])
        self.add_channel = add_channel
        self.live_generation = live_generation
        self.preprocess = preprocess
        self.n_mels = n_mels
        np.random.shuffle(self.indices)


    def __len__(self):
        return math.floor(len(self.x)/self.batch_size)

    def __getitem__(self, index):
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y = self.y[inds]

        return np.array([np.expand_dims(_get_spec(x, live_generation=self.live_generation, 
                            preprocess = self.preprocess, 
                            n_mels = self.n_mels), -1) 
                            if self.add_channel
                            else _get_spec(x, live_generation=self.live_generation) 
                            for x in batch_x]), np.array(batch_y)

    def on_epoch_end(self):
        self.indices = np.random.shuffle(self.indices)