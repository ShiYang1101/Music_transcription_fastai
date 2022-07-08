import librosa
import glob
import os 
import math
import numpy as np
import matplotlib.pyplot as plt
import random


def add_noise(_spec, plot = False, seed = 42):
    '''
    >>> test = np.ndarray((3, 5))
    >>> add_noise(test).shape
    (3, 5)
    '''
    np.random.seed(seed)
    output = _spec + np.random.normal(0, max(_spec) * 0.2, size = _spec.shape)
    if plot == True:
        plt.plot(_spec)
        plt.plot(range(len(output)), output)
    return output

def mask_spec(arr, inplace = False):
    '''
    Function masking the spectrogram, randomly choose 2 to 3 startin point in spectrogram
    numpy array, and setting random duration after it to magnitude 0. Applies both 
    to the time and frequnecy dimension.
    
    Input: numpy.ndarray
    Output: numpy.ndarray

    
    '''
    if not isinstance(arr, np.ndarray):
        raise TypeError("The input type should be spectrogram represented in numpy array!")
    loop = random.randint(1, 2)
    tmp = arr.copy()
    for i in range(loop):
        start = random.randint(0, arr.shape[1])
        duration = random.randint(25, 60)
        if inplace == True:
            arr[:, start:start + duration] = 0
        else:
            tmp[:, start:start+duration] = 0
    freq_loop = random.randint(1, 3)
    for freq in range(freq_loop):
        start = random.randint(0, arr.shape[0])
        duration = random.randint(25, 60)
        if inplace == True:
            arr[start:start + duration, :] = 0
        else:
            tmp[start:start + duration, :] = 0

    return None if inplace == True else tmp

def generate_spec(path, sr = None, full_path = False, noise = True, 
                    **kwargs):
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
    if noise == True:
        file = add_noise(file)
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

def path_to_preprocessing(path, noise_prob = 0.3, mask_prob = 0.3):
    _spec = generate_spec(path)
    if random.random() < noise_prob:
        _spec = add_noise(_spec)
    if random.random() < mask_prob:
        _spec = mask_spec(_spec)
    return _spec
