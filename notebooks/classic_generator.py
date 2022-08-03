import glob
import os
import math

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from librosa import display
from tensorflow.keras.utils import Sequence

from spectrogram_class import spectrogram

path_to_notebooks = os.path.dirname(__file__)
path_to_data = glob.glob('../**/musicnet/', recursive=True)[0]
_full_path_to_data = os.path.join(path_to_notebooks, path_to_data)
_train_label_path = os.path.join(path_to_notebooks, path_to_data, 'train_labels/')

_df_list = []
for _labels_csv in os.listdir(_train_label_path):
    _df_list.append(pd.read_csv(os.path.join(_train_label_path, _labels_csv)))
_train_label_df = pd.concat(_df_list, ignore_index=True)

_instrument_list = sorted(_train_label_df['instrument'].unique())
_note_list = sorted(_train_label_df['note'].unique())

_sr = 44100

def classic_train_generator(path, **kwargs):
    spec = spectrogram(path, trunc_off = True, **kwargs)
    return spec.spec

def truncate_spec(arr, max_len):
    if arr.shape[1] < max_len:
        to_pad = max_len - arr.shape[1]
        arr = np.pad(arr, ((0, 0), (0, to_pad)))
        return arr
    elif arr.shape[1] > max_len:
        arr = arr[:,:max_len]
        return arr
    else:
        return arr

def spec_scaler(arr):
    return (arr - arr.mean())/arr.std()

def get_full_path(path, mode):
    return os.path.join(_full_path_to_data, 
                f"{mode}_data", path.rsplit('/')[-1].rsplit('.')[0] + '.wav')


def _instrument_label_generator(df_path, ins, time_len, mode, sr = 44100):
    _inner_df = pd.read_csv(df_path)
    _train_path = os.path.join(df_path.rsplit('/', maxsplit = 1)[0] + '/', 
                f"../{mode}_data", df_path.rsplit('/')[-1].rsplit('.')[0] + '.wav')
    num_dur_sr = int(librosa.get_duration(filename = _train_path, sr = 44100) * _sr)
    _inner_df = _inner_df[_inner_df['instrument'] == ins]
    _inner_arr = np.zeros((len(_note_list), time_len))
    for _rows in _inner_df.iterrows():
        _inner_arr[_note_list.index(_rows[1]['note']), 
                        int(_rows[1]['start_time']/num_dur_sr * time_len) : \
                                        int(_rows[1]['end_time']/num_dur_sr * time_len)] = 1
    return _inner_arr



class classic_generator(Sequence):
    
    def __init__(self, mode = 'train', batch_size = 32):
        self.x_path = os.path.join(path_to_notebooks, path_to_data, f"{mode}_data/")
        self.y_path = os.path.join(path_to_notebooks, path_to_data, f"{mode}_labels/")
        self.x = pd.Series([get_full_path(x, mode = mode) for x in 
                                    os.listdir(self.x_path)])
        self.y = pd.Series([os.path.join(path_to_notebooks, path_to_data, f"{mode}_labels/", label) 
                        for label in os.listdir(self.y_path)])
        self.batch_size = batch_size
        self.mode = mode
        self.indices = np.arange(len(self.x))
        np.random.shuffle(self.indices)


    def __len__(self):
        return math.floor(len(self.x)/self.batch_size)

    def __getitem__(self, index):
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y = self.y[inds]
        batch_x_spec = [spec_scaler(classic_train_generator(x)) for x in batch_x]
        batch_time = [x.shape[1] for x in batch_x_spec]
        max_batch_time = max([x.shape[1] for x in batch_x_spec])
        batch_x_spec = [truncate_spec(x, max_batch_time).T for x in batch_x_spec]

        return np.array(batch_x_spec), \
                            {f"instrument_{ins}": np.array([truncate_spec(_instrument_label_generator(label, ins, time, 
                                                mode= self.mode), max_batch_time).T for label, time 
                                                        in zip(batch_y, batch_time)])
                                                            for ins in _instrument_list} 

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
