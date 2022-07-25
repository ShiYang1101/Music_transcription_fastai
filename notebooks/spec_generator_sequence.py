import tensorflow as tf
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import OneHotEncoder
import math
import os
import glob
from spectrogram_class import spectrogram
import pandas as pd
import numpy as np

dir_path = os.path.dirname(__file__)
_path_to_npy = glob.glob('../data/**[!MACOSC]/*OrchideaSOL2020/', recursive=True)[0]
meta_df = pd.read_csv('../data/OrchideaSOL_metadata.csv')
_onehot = OneHotEncoder(sparse=False)
_onehot.fit(meta_df[['Instrument (in full)']])

def _get_spec(path, test_verbose = False):
    path = os.path.join(dir_path, _path_to_npy, path)
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
        spec = spectrogram(path)
    # if preprocess = True:
    #     spec.
    # spec.spec = np.expand_dims(spec.spec, -1)
    return spec.spec

class spec_generator(Sequence):

    def __init__(self, df, batch_size):
        self.x = df['Path'].array
        self.y = _onehot.transform(df[['Instrument (in full)']])
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