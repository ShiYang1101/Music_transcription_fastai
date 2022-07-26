import librosa
from librosa import display
import pandas as pd
from spectrogram_class import spectrogram
import numpy as np
import glob
import tensorflow as tf
from tensorflow.keras.utils import Sequence

path_to_data = glob.glob('../**/musicnet/', recursive=True)[0]
print('HIT', path_to_data)

def classic_train_generator(path, **kwargs):
    spec = spectrogram(path, trunc_off = True, **kwargs)
    return spec.spec

class classic_generator(mode = 'train'):
    
    def __init__()