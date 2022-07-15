from glob import glob
import numpy as np
import os
import glob
import random
from spectrogram_class import spectrogram

dir_path = os.path.dirname(__file__)
_path_to_npy = glob.glob('../data/**[!MACOSC]/*OrchideaSOL2020/', recursive=True)[0]
print(_path_to_npy)

def gen(train_df, noise_prob = 0.3, mask_prob = 0.3, preprocess = False, 
                return_class = False):
    df = train_df.copy()
    while True: 
        tmp_df = df.sample(1)
        path = os.path.join(dir_path, _path_to_npy, tmp_df['Path'].values[0])
        try:
            spec = np.load(os.path.splitext(path)[0] + '.npy')
        except:
            spec = spectrogram(path)
        # if preprocess = True:
        #     spec.
        spec.spec = np.expand_dims(spec.spec, -1)
        yield (spec.spec if return_class == False else spec, 
        np.reshape(tmp_df[list(set(df.columns) - set(['Path']))].values, (16,)))
    
def gen_eval(test_df, noise_prob = 0.3, mask_prob = 0.3):
    df = test_df.copy()
    while True: 
        tmp_df = df.sample(1)
        path = os.path.join(dir_path, _path_to_npy, tmp_df['Path'].values[0])
        try:
            spec = np.load(os.path.splitext(path)[0] + '.npy')
        except:
            spec = spectrogram(path)
        spec.spec = np.expand_dims(spec.spec, -1)
        yield (spec.spec, np.reshape(tmp_df[list(set(df.columns) - set(['Path']))].values, (16,)))