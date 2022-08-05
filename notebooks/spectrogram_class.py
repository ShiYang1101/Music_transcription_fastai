import glob
import math
import os
import random
import string
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as nps
import warnings

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

class spectrogram(object):
    '''
    Class object for spectrogram or augmentation, utilized librosa.
    '''


    def __init__(self, input, hop_length = 3500, n_fft = 4096, n_mels = 64, 
                preprocess = True, trunc_length = 500, trunc_off = False):
        '''
        Initialized spcectrogram instance, utilized Librosa modules for generating spectrogram
        in numpy's ndarray format. One of the path to audio files or ndarray of 2 dimensions 
        should be provided. Defaulted to use native sampling rate of the audio file.
        
        Input: str (Path to audio file)/ np.ndarray (2 dimensions)

        Optional arguments:
        hop_length: Number of sample to skip for generating short-time Fourier Transformation.
        win_length: Window length to calculate short-time Fourier Transformation, in the units
                    of sample.
        n_fft: Magnitude of frequency bins after Fourier transform. The number of frequency bins
               in the output will be n_fft//2 + 1.
        n_mels: int, number of frequency bins after mel spectrogram generation
        preprocess: Boolean, whether to apply adding noise, masking and shifting 
                    audio files generated
        trunc_off: Boolean, whether to turn off truncateion of spectrogram
        trun_length: int, length of spectrogram time dimension to be truncated.
        
        '''
        if isinstance(input, str):
            self.n_mels = n_mels
            try:
                # Generating signal from path
                self.signal, self.sr = librosa.load(input, sr = None)
            except:
                # Getting the path to audio file
                dirpath = os.path.dirname(__file__)
                rel_path = glob.glob(os.path.join(dirpath, '..', 'data/**[!MACOSC]/*OrchideaSOL2020/'), 
                                recursive=True)[0]

                # Generatinf signal
                self.signal, self.sr = librosa.load(rel_path + input, sr = None)

            self.hop = hop_length
            self.n_fft = n_fft
            if preprocess == True:
                # Adding noise to the signal
                self.add_noise() 
            # Generate mel spectrogram from signal
            self.generate_spec(input, hop_length = hop_length, 
                                        n_fft = n_fft)
            if preprocess == True:
                # Applyting masking and shifting to spectrogram
                self.mask_spec()
                self.shift_spec()
        elif isinstance(input, np.ndarray):
            self.n_mels = n_mels
            self.spec = input
        if not trunc_off:
            self.truncate_spec(trunc_length)

        # Sanity check that the output is in desired shape 
        assert isinstance(self.spec, np.ndarray), 'The spectrogram generate is not in the form of np.array!'
        assert self.spec.ndim == 2, f"The spectrogram is not a 2 dimensional np array! It is a {self.spec.shape} array."

    def add_noise(self, noise_factor = 0.05, plot = False):
        '''
        Only to be used within the generate_spec method under spectrogram class.
        Method for adding noise to signals. By default, the noise added
        to the signal will default to be the normal distribution with standard deviation
        of 20 percent of maximum magnitude.

        Input: Instance of spectrogram class
        >>> test = np.ndarray((3, 5))
        >>> add_noise(test).shape
        (3, 5)
        '''
        self.signal = self.signal + np.random.normal(0, max(self.signal) * noise_factor, size = self.signal.shape)

    def mask_spec(self, inplace = False):
        '''
        Function masking the spectrogram, randomly choose 2 to 3 startin point in spectrogram
        numpy array, and setting random duration after it to magnitude 0. Applies both 
        to the time and frequnecy dimension.
        
        Input: numpy.ndarray
        Output: numpy.ndself.specay

        
        '''
        # Generating number of masking number
        loop = random.randint(1, 2)
        tmp = self.spec.copy()

        # Masking the time dimension 
        for i in range(loop):
            start = random.randint(0, self.spec.shape[1])
            duration = random.randint(25, 60)
            self.spec[:, start:start + duration] = 0
        freq_loop = random.randint(1, 2)
         
        # Masking the frequency dimension
        for freq in range(freq_loop):
            start = random.randint(0, self.spec.shape[0])
            duration = random.randint(2, 10)
            self.spec[start:start + duration, :] = 0

    def generate_spec(self, sr = None, full_path = False, noise = True, 
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
        self.spec = librosa.feature.melspectrogram(self.signal, n_mels = self.n_mels, hop_length=self.hop, n_fft=self.n_fft)
        if self.spec.ndim == 3:
            self.spec = np.reshape(self.spec, self.spec.shape[:2])

    def truncate_spec(self, max_len):
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
        if self.spec.shape[1] < max_len:
            to_pad = max_len - self.spec.shape[1]
            left_pad = math.floor(to_pad/2)
            right_pad = to_pad - left_pad 
            self.spec = np.pad(self.spec, ((0, 0), (left_pad, right_pad)))
        elif self.spec.shape[1] > max_len:
            self.spec = self.spec[:,:max_len]
    


    def preprocess(self, noise_prob = 0.3, mask_prob = 0.3):
        '''
        Method for preprocessing of spectrogram, produce masking, shifting and scaling magnitude 
        for signals.'''
        if random.random() < noise_prob:
            self.add_noise()
        if random.random() < mask_prob:
            self.mask_spec()

    def plot_spec(self, db_off = False, **kwargs):
        '''
        Support function for plotting the spectrogram
        '''
        if self.spec.ndim > 2:
            if not db_off:
                ax = plt.subplot()
            im = librosa.display.specshow(librosa.amplitude_to_db(np.reshape(self.spec, self.spec.shape[:2])), 
                                x_axis='s', sr = self.sr, 
                                y_axis= 'mel', hop_length=self.hop, n_fft=self.n_fft, 
                                **kwargs)
            if not db_off:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax = cax, format="%+2.f dB")
            plt.show()
        else:
            if not db_off:
                ax = plt.subplot()
            im = librosa.display.specshow(librosa.amplitude_to_db(self.spec), x_axis='s', sr = self.sr, 
                                    y_axis= 'mel', hop_length=self.hop, n_fft=self.n_fft, 
                                    **kwargs)
            if not db_off:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax = cax, format="%+2.f dB")
            plt.show()
    

    def shift_spec(self, max_sec = 0.5):
        tmp = self.spec.copy()
        roll_window = int(np.random.uniform(int(self.sr* max_sec/self.hop), 
                                        int(self.sr/self.hop)))
        tmp = np.roll(tmp, roll_window, axis = 1)
        self.spec = tmp

if __name__ == '__main__':
    test = spectrogram('PluckedStrings/Harp/pizzicato_bartok/Hp-pizz_bartok-G3-ff-N-N.wav')
    print(test.spec.shape)
    test.preprocess()
    print(test.spec.shape, 'HIT')
    test.plot_spec()
