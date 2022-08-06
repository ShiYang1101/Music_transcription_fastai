Data location: AWS S3 bucket
[s3://shiyanglee-bstn-bucket](https://s3.console.aws.amazon.com/s3/buckets/shiyanglee-bstn-bucket?region=eu-west-2&tab=objects)

>> Note that the data folder should be located in the parent folder of current **notebooks** folder for the data generator to works!

Notebooks order:
1. EDA
1. music_transcription_class
1. music_transcription_2conv
1. music_transcription_RNN
1. classic_transcription

Notebook description:

|Notebook|Description|
|---|---|
|EDA|Preliminary cleaning/EDA for the 2 datasets: OrchideaSOL and MusicNet|
|music_trancription_class|Contains demonstration of preprocessing for spectrogram class (defined in spectrogram_class.py) and baseline CNN model for OrchideaSOL instrument classification|
|music_transcription_2conv| Deeper CNN model with 2 convolutional layer for OrchideaSOL|
|music_transcription_RNN| Utilized RNN LSTM model for OrchideaSOL instrument and pitch classification|
|classic_transcription| Sequential RNN LSTM model music transcription for MusicNet dataset|

py files description:
|Py file|Description|
|---|---|
|spectrogram_class|Custom class for sprectrogram generation and data augmentation|
|spec_generator_sequence|Module build on tensorflow keras Sequential class for data generation of OrchideaSOL data, in the form of spectrogram and instrument label|
|spec_generator_sequence_multilabel|Generator for tensorflow model, with label of instrument and notes|
|classic_generator|Generator for MusicNet dataset, includes spectrogram generation, and label of timestep and notes 2d np array, for each instruments available|
|wav_converter|Convert audio files into npy files with specified librosa spectrogram parameters, saved under data folder of parent directory|

