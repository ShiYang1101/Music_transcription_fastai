import glob
import os
import math

import librosa
from matplotlib.pyplot import isinteractive
import numpy as np
import pandas as pd
import tensorflow as tf
from librosa import display
from tensorflow.keras.utils import Sequence

from spectrogram_class import spectrogram

'''
This model is a class module for generating training and evaluating dataset 
for tensorflow. We will mainly working on the Seqence class by tensorflow keras.
Note that the musicnet data have to be is the subdirectory of the parent folder
of this file.
'''

# Getting the absolute path to this file
path_to_notebooks = os.path.dirname(__file__)

# Getting the relative path of the dataset from this py file
path_to_data = glob.glob('../**/musicnet/', recursive=True)[0]

# Getting the full path to the MusicNet dataset
_full_path_to_data = os.path.join(path_to_notebooks, path_to_data)

# Getting the absolute path to the training labels
_train_label_path = os.path.join(path_to_notebooks, path_to_data, 'train_labels/')

# In order to get the possible instrument and note for classfication, 
# we will first be joining the csv files for all of training label csv
# and get the unique value for both
_df_list = []

# Iterate over all csv files
for _labels_csv in os.listdir(_train_label_path):
    # Appending all csv file to a list
    _df_list.append(pd.read_csv(os.path.join(_train_label_path, _labels_csv)))

# Getting the full training dataframe using pd.concat
_train_label_df = pd.concat(_df_list, ignore_index=True)


# Getting the list of unique instruments and notes
_instrument_list = sorted(_train_label_df['instrument'].unique())
_note_list = sorted(_train_label_df['note'].unique())

_sr = 44100

def classic_train_generator(path, **kwargs):
    '''
    Function for generating spectrogram by a given path to the audio file, 
    the output is a mel-scaled spectrogram, with time and frequency bins, 
    represented in np.arrays.
    
    Input: str, path to audio file
    Output: np.array, 2dimensional
    
    Optional:
    **kwargs: Parameters to be passed in the spectrogram class initialization.
    '''
    spec = spectrogram(path, hop_length = 4000, trunc_off = True, **kwargs)
    return spec.spec

def truncate_spec(arr, max_len):
    '''
    Supporting function for truncating/trimming/padding spectrogram, represented 
    in np.array. The spectrogram will be padded with zeros, increasing the time
    dimension if the max_len is larger than input spectrogram. Otherwise,
    the spectrogram will be truncated.
    
    Input:
    arr: np.array, 2 dimensional, the first dimension should be the frequency bins,
        whereas the second dimension correspond to the tim bins.
    max_len: int, the desired length of np.array in the time dimension
    
    Output: np.array, 2 dimensional with same frequency bins and time dimension
            with the max_len provided
    '''
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
    '''
    Support function for standard scaling the np.array
    
    Input:
    arr: np.array
    
    Output: np.array.
    '''
    if not isinstance(arr, (type(np.array), (np.ndarray))):
        raise TypeError('The input type is not a np.array!')
    return (arr - arr.mean())/arr.std()

def get_full_path(path, mode):
    '''
    Support function to get the full path to the audio files, by the labels
    csv path.

    Input:
    path: str, path to the label csv file.
    mode: str, 'train' or 'test', where the input path belongs to training
            or testing dataset
    
    Output: str, full path to the corresponding audio files.
    '''
    return os.path.join(_full_path_to_data, 
                f"{mode}_data", path.rsplit('/')[-1].rsplit('.')[0] + '.wav')


def _instrument_label_generator(df_path, ins, time_len, mode, sr = 44100):
    '''
    Function generating the labels to the desired format. The output is the 
    2 dimensional np.array with the label for notes in the time slices. The 
    first dimension is the note ranges, wheareas the second dimension 
    represents the time dimension. The function will only generate label for
    one instrument.
    
    Input:
    df_path: str, the path to the labels
    ins: int, the code representing the instrument
    time_len: int, the number of time slices for the corresponding generated
                spectrogram
    mode: str, 'train' or 'test', the indicator for the data type for the label
    sr: int, the sample rate of the correponding audio file, used to convert the
        labels in csv files in to corresponding position in np.array label output.
        
    Output: np.array, 2 dimensional. The first dimension correspond to the note 
            rages, whereas the second dimensio corresponds to the time dimensional
            with the same length to the corresponding generated spectrogram
    '''

    # Converting the label csv file in to a pd dataframe
    _inner_df = pd.read_csv(df_path)

    # Getting the path to the corresponding audio file
    _train_path = os.path.join(df_path.rsplit('/', maxsplit = 1)[0] + '/', 
                f"../{mode}_data", df_path.rsplit('/')[-1].rsplit('.')[0] + '.wav')

    # Getting the number of duration of the audio file
    num_dur_sr = int(librosa.get_duration(filename = _train_path, sr = 44100) * _sr)

    # Slicing the label dataframe to only include the desired instrument
    _inner_df = _inner_df[_inner_df['instrument'] == ins]
    _inner_arr = np.zeros((len(_note_list), time_len))

    # Iterating over the label dataframe, to fill the output np array
    # with with the target (1) for corresponding time slices and notes
    for _rows in _inner_df.iterrows():
        _inner_arr[_note_list.index(_rows[1]['note']), 
                        int(_rows[1]['start_time']/num_dur_sr * time_len) : \
                                        int(_rows[1]['end_time']/num_dur_sr * time_len)] = 1
    return _inner_arr



class classic_generator(Sequence):

    '''
    Sub-class of tensorflow keras Sequence class. Act as a dataset generator
    for training and evaluating purpose of MusicNet dataset. 
    '''
    
    def __init__(self, mode = 'train', batch_size = 32):
        '''
        Initialization for classic_generator class. The instance of this class
        act as a dataset generator for tensorflow keras models. 
        
        Input: 
        mode: str, 'train' or 'test', used to determine the path to audio and label files
        batch_size: int, defaulted to 32, number of sample to generate in a batch
        '''

        # Getting the absolute path to the audio files
        self.x_path = os.path.join(path_to_notebooks, path_to_data, f"{mode}_data/")
        # Getting the absolute path to the label files
        self.y_path = os.path.join(path_to_notebooks, path_to_data, f"{mode}_labels/")

        # pd Series for the available audio files
        self.x = pd.Series([get_full_path(x, mode = mode) for x in 
                                    os.listdir(self.x_path)])

        # pd.Series for the available label files
        self.y = pd.Series([os.path.join(path_to_notebooks, path_to_data, f"{mode}_labels/", label) 
                        for label in os.listdir(self.y_path)])

        self.batch_size = batch_size
        self.mode = mode

        # Assingning indices for the audio and labels files to be used
        self.indices = np.arange(len(self.x))

        # Initial shuffling for the data
        np.random.shuffle(self.indices)


    def __len__(self):
        '''
        Supporting functing to the the number of steps per epoch
        '''
        return math.floor(len(self.x)/self.batch_size)

    def __getitem__(self, index):
        '''
        Function for generating feature/labels pair

        Input:
        index: int, batch number

        Output: tuple, first element correspond to the array of spectrogram, according
                to batch number.
                Second element correspond to the dictionary of output labels, represented
                in array of time slices and note range

        '''
        # Generating the indices to be used in the batch
        inds = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Getting the corresponging path to audio and label files
        batch_x = self.x[inds]
        batch_y = self.y[inds]

        # Generate spectrogram array
        batch_x_spec = [spec_scaler(classic_train_generator(x)) for x in batch_x]

        # Getting the number of time slices for each generated spectrogram in a list
        batch_time = [x.shape[1] for x in batch_x_spec]

        # Getting the maximum time slices in the batch for padding purpose
        max_batch_time = max([x.shape[1] for x in batch_x_spec])

        # Performing final truncation and dimension modification for spectrogram arrays
        batch_x_spec = [np.expand_dims(truncate_spec(x, max_batch_time).T, -1) 
                                                        for x in batch_x_spec]

        # The output correspond to the spectrograms generated, and a dictionary with 
        # instruments key. Each values in the instrument is the array of labels arrays
        # with the, corresponding to the batch size
        return np.array(batch_x_spec), \
                            {f"instrument_{ins}": np.array([truncate_spec(_instrument_label_generator(label, ins, time, 
                                                mode= self.mode), max_batch_time).T for label, time 
                                                        in zip(batch_y, batch_time)])
                                                            for ins in _instrument_list} 

    def on_epoch_end(self):
        # Shuffling the indices to be used in the next epoch
        np.random.shuffle(self.indices)
