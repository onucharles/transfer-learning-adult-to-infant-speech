"""
This scripts renames the wav files in the chillanto database to a format that can be used by our hashing algorithm to
reliably split patients into training, validation and test set, such that all samples from one patient lie in only one set.

In this example, "0021" is the patient id while "003004" is the sample id.
Chillanto format: 0021003004
Destination format: 003004_nohash_0021

python wav_handling/rename_chillanto_wavs.py \
--source_dir=/mnt/hdd/Datasets/chill_deaf8k \
--destination_dir=/mnt/hdd/Datasets/chill_deaf8k_renamed \

"""
import sys
import platform

##project_root = r'D:\Users\Charley\Documents\Esperanza\ml_projects\ubenwa-transfer-learning'\
#               if platform.system() == 'Windows' else '~/ubenwa-transfer-learning/'
#sys.path.insert(0, project_root)

import argparse
import os
from shutil import copyfile, copy2
#from utils.ioutils import create_folder
import os
import json
import time


def create_folder(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        print("created directory: " + str(newpath))


def get_new_name(file_name):
    if not file_name:
        raise ValueError('empty file name was passed!')

    patient_id = file_name[:4]
    sample_id = file_name[4:]

    new_file_name = patient_id + '_nohash_' + sample_id

    return new_file_name

def main():
    source_dir = FLAGS.source_dir
    destination_dir = FLAGS.destination_dir

    # check if source_dir exists. (if no: throw exception)
    if not os.path.exists(source_dir) or not os.path.isdir(source_dir):
        raise ValueError('source_dir does not exist or is not a directory!')

    # check if dest_dir exists (if yes: throw exception)
    if os.path.exists(destination_dir):
        raise ValueError('destination_dir already exists!')

    # load files in source_dir and save to destination_dir
    dir_count = 0
    total_file_count = 0
    for dir in os.listdir(source_dir):
        cur_sub_dir = os.path.join(source_dir, dir)
        if not os.path.isdir(cur_sub_dir):
            print("Skipped one file which is not a directory.")
            continue
        dest_sub_dir = os.path.join(destination_dir, dir)
        create_folder(dest_sub_dir)

        file_count = 0
        for file in os.listdir(cur_sub_dir):
            if not file.endswith(".wav"):
                continue
            cur_file_name = os.path.join(cur_sub_dir, file)
            new_file_name = os.path.join(dest_sub_dir, get_new_name(file))
            copy2(cur_file_name, new_file_name)
            file_count += 1

        print("Renamed {0} wav files in sub-directory '{1}'"
              .format(file_count, dir))
        dir_count += 1
        total_file_count += file_count

    print("Complete!\nRenamed a total of {0} wav files across {1} subdirectories"
          .format(total_file_count, dir_count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_dir',
        type=str,
        default=None,
        # required=True,
        help='Location of source wav files.')
    parser.add_argument(
        '--destination_dir',
        type=str,
        default=None,
        # required=True,
        help='Where to save renamed wav files.')

    FLAGS, unparsed = parser.parse_known_args()
    main()

# import librosa
# def test_renamed_file():
#     file1 = '/mnt/hdd/Datasets/chillanto-8k-16bit/asphyxia/0063002030.wav'
#     file2 = '/mnt/hdd/Datasets/chillanto-8k-16bit-renamed/asphyxia/0063_nohash_002030.wav'
#
#     y1, sr1 = librosa.load(file1)
#     y2, sr2 = librosa.load(file2)
#     assert (y1 == y2).all()
#     assert sr1 == sr2
