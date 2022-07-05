import spectrogram as sp
import sys
import glob
import numpy as np
import os

def main():
    print('Converting all wav files in subdirectories to sfft numpy files....')
    wav_list = glob.glob('../data/**/*.wav', recursive= True)
    convert_num = 0
    fail_num = 0
    overwrite_num = 0
    for path in wav_list:
        try:
            spec = sp.generate_spec(path, full_path = True, 
                                    hop_length = int(sys.argv[1]), 
                                    win_length = int(sys.argv[2]), 
                                    n_fft = int(sys.argv[3]))
            spec = sp.truncate_spec(spec, int(sys.argv[4]))
            outfile = path.rsplit('.', 1)[0]
            if os.path.exists(outfile + '.npy'):
                overwrite_num += 1
            else:
                convert_num += 1
            np.save(outfile, spec)
        except Exception as e:
            print(e)
            fail_num += 1
            pass
    print(f"Successfully converted {convert_num} files, ")
    print(f"Fail to convert {fail_num} files, ")
    print(f"Overwriten {overwrite_num} files.")
    return 0

if __name__ == '__main__':
    main()