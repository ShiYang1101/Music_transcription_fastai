import tensorflow as tf
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import OneHotEncoder
import math
import os
import glob
from spectrogram_class import spectrogram
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

dir_path = os.path.dirname(__file__)
_path_to_npy = glob.glob('../data/**[!MACOSC]/*OrchideaSOL2020/', recursive=True)[0]
_meta_df = pd.read_csv('../data/OrchideaSOL_metadata.csv')
_onehot = OneHotEncoder(sparse=False)
_onehot.fit(_meta_df[['Instrument (in full)']])


def _get_multilabel(df):
    df['_multilabel'] = [[i, str(j)] for i, j in zip(df['Instrument (in full)'], 
                                df['Pitch ID (if applicable)'])]

_get_multilabel(_meta_df)
_multi = MultiLabelBinarizer()
_multi.fit(_meta_df['_multilabel'])

_instrument_multi = OneHotEncoder(sparse=False)
_instrument_multi.fit(_meta_df[['Instrument (in full)']])

_pitch_multi = OneHotEncoder(sparse= False)
_pitch_multi.fit(_meta_df[['Pitch ID (if applicable)']].astype(str))

def _get_spec(path, test_verbose = False):
    path = os.path.join(dir_path, _path_to_npy, path)
    try:
        if test_verbose:
            print('HIT')
        return np.expand_dims(np.transpose(np.load(os.path.splitext(path)[0] + \
                                '.npy', allow_pickle=True)), -1)
        if test_verbose:
            print("SUCCESS")
    except:
        if test_verbose:
            print("FAILED")
        spec = spectrogram(path)
    # if preprocess = True:
    #     spec.
    spec.spec = np.transpose(spec.spec)
    spec.spec = np.expand_dims(spec.spec, -1)
    return spec.spec

class spec_generator_multi(Sequence):

    def __init__(self, df, batch_size):
        _get_multilabel(df)
        self.x = df['Path'].array
        self.y = _multi.transform(df['_multilabel']) 
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])
        np.random.shuffle(self.indices)


    def __len__(self):
        return math.floor(len(self.x)/self.batch_size)

    def __getitem__(self, index):
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y = self.y[inds]

        return np.array([_get_spec(x) for x in batch_x]), np.array(batch_y)

    def on_epoch_end(self):
        self.indices = np.random.shuffle(self.indices)


class spec_generator_multioutput(Sequence):

    def __init__(self, df, batch_size):
        self.x = df['Path'].array
        self.y_instrument = _instrument_multi.transform(df[['Instrument (in full)']]) 
        self.y_pitch = _pitch_multi.transform(df[['Pitch ID (if applicable)']].astype(str)) 
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])
        np.random.shuffle(self.indices)


    def __len__(self):
        return math.floor(len(self.x)/self.batch_size)

    def __getitem__(self, index):
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y_instrument = self.y_instrument[inds]
        batch_y_pitch = self.y_pitch[inds]

        return np.array([_get_spec(x) for x in batch_x]), \
                            {'out1': np.array(batch_y_instrument), 
                                'out2': np.array(batch_y_pitch)}

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
