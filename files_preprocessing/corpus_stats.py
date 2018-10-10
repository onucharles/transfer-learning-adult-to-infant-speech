"""
Prints relevant statistics about audio files in a corpus. Eg. sampling rates.


python corpus_stats.py --source_dir=/mnt/hdd/Datasets/chillanto/
"""
import os.path
import argparse
import soundfile as sf

FLAGS = None

def print_sampling_rates():
    source_dir = FLAGS.source_dir

    # check if source_dir exists.
    if not os.path.exists(source_dir) or not os.path.isdir(source_dir):
        raise ValueError('source_dir does not exist or is not a directory!')

    dir_count = 0
    total_file_count = 0

    # go through sub-folders in source directories
    for dir in os.listdir(source_dir):
        cur_sub_dir = os.path.join(source_dir, dir)
        if not os.path.isdir(cur_sub_dir):
            print("Skipped one file which is not a directory.")
            continue

        file_count = 0

        # go through files in current sub-folder.
        sampling_freqs = {}
        print("Looking in directory '{0}'...".format(dir))
        for file in os.listdir(cur_sub_dir):
            if not file.endswith(".wav"):
                continue
            cur_file_name = os.path.join(cur_sub_dir, file)
            y, sr = sf.read(cur_file_name, dtype='float32')

            if sr in sampling_freqs:
                sampling_freqs[sr] += 1
            else:
                sampling_freqs[sr] = 1
            # print('Sampling frequency of "{0}" is "{1}"'.format(file, sr))

            file_count += 1
        print(sampling_freqs)
        dir_count += 1
        total_file_count += file_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_dir',
        type=str,
        default=None,
        help='Location of source wav files.')

    FLAGS, unparsed = parser.parse_known_args()
    print_sampling_rates()