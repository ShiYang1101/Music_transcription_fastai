import librosa
import glob
import os 
import math
import numpy as np

def generate_spec(path, sr = None, full_path = False, **kwargs):
    '''
    Generate spectrogram's numpy nd.array base on directory.
    The first dimension contains the frequency bins, whereas the second
    dimension represents the windows bin.

    The dtype of the numpy array is of complex64 type. The magniudes
    represent the magnitude of the frequency bins, and the angles represent
    the phase of corresponding frequency.

    Input: String, path to audio file.
    Output: 2 dimensions nd.array.
    '''
    rel_path = glob.glob('../data/**[!MACOSC]/*OrchideaSOL2020/', recursive=True)[0]
    if full_path == False:
        true_path = rel_path + path
    else:
        true_path = path
    file, sr = librosa.load(true_path, sr=sr)
    y = librosa.stft(file, **kwargs)
    return np.abs(y)

def truncate_spec(sig, max_len):
    '''
    A function that truncate the spectrogram numpy array to desired shape.
    Assumed that the first dimension represent the frequency bins, and the 
    second dimension represents the time window bins. The function only 
    truncates in the time dimensions, with padding of 0s.

    Input: np.ndarray with 2 dimensions.
    Output: np.ndarray with 2 dimension.

    >>> import numpy as np
    >>> test = np.ones((2, 3))
    >>> truncate_spec(test, 5)
    array([[0., 1., 1., 1., 0.],
           [0., 1., 1., 1., 0.]])
    >>> truncate_spec(test, 2)
    array([[1., 1.],
           [1., 1.]])
    '''
    if sig.shape[1] < max_len:
        to_pad = max_len - sig.shape[1]
        left_pad = math.floor(to_pad/2)
        right_pad = to_pad - left_pad 
        return np.pad(sig, ((0, 0), (left_pad, right_pad)))
    elif sig.shape[1] > max_len:
        return sig[:,:max_len]
    else:
        return sig


def add_noise(spec):
    std = spec.std()
    return spec + np.random.normal(0, std, size = spec.shape)
        