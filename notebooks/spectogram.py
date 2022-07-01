import librosa
import glob
import os 

def generate_spec(path):
    file, sr = librosa.load(path)
    y = librosa.stft(file)
    return y

def get_spec(df):
    path = glob.glob('**/_OrchideaSOL2020_release', recursive = True)
    if isinstance(path, list):
        path = path[0]
    for i in df.Path:
        file_path = path + i

        
        
        