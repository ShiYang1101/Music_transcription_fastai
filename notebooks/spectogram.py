import librosa
import glob
import os 
import math
import numpy as np

def generate_spec(path):
    file, sr = librosa.load(path)
    y = librosa.stft(file)
    return y

def truncate_spec(sig, max_len):
    if sig.shape[0] < max_len:
        to_pad = max_len - sig.shape[0]
        left_pad = math.floor(to_pad/2)
        right_pad = math.ceil(to_pad/2)
        return np.pad(sig, ((0, 0), (left_pad, right_pad)))


def get_spec(df):
    path = glob.glob('**/_OrchideaSOL2020_release', recursive = True)
    if isinstance(path, list):
        path = path[0]
    for i in df.Path:
        file_path = path + i

        
        
        