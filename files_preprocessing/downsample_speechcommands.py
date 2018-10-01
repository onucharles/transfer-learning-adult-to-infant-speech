"""
Downsamples wav files within a source directory to a specified sampling rate.
Files are saved to a specified destination.

Important:
Code assumes that wav files are stored within sub-folders in the source directory.
The destination directory is then organised similarly.
e.g
If source is organised as:
-> source_dir/
    -> pos/
        -> 001.wav
        -> 002.wav
    -> neg/
        ->ggg.wav

Destination will be organised as (where wav files have new sampling rate):
-> dest_dir/
    -> pos/
        -> 001.wav
        -> 002.wav
    -> neg/
        -> ggg.wav

Script:
python downsample_speechcommands.py --source_dir='/mnt/hdd/Datasets/chillanto/' \
--destination_dir='/mnt/hdd/Datasets/chillanto-8k-16bit/' --resample_rate=8000

"""

import os
import argparse
import librosa
import soundfile as sf

FLAGS = None

def create_folder(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)

def downsample_and_save():
    """
    Iterate through sub-folders in source_dir and save to an equivalent sub-folder in destination_dir.
    :return:
    """

    source_dir = FLAGS.source_dir
    destination_dir = FLAGS.destination_dir
    new_sr = FLAGS.resample_rate

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
            #y, sr = librosa.load(cur_file_name, sr=new_sr)
            y, sr = sf.read(cur_file_name, dtype='float32')
            y = y.T     # sf reads in (nb_samples, nb_channels) whereas librosa expects transpose.
            y_resampled = librosa.resample(y, sr, new_sr)

            # save new wav file.
            new_file_name = os.path.join(dest_sub_dir, file)
            #librosa.output.write_wav(new_file_name, y_resampled, new_sr)
            sf.write(new_file_name, y_resampled.T, new_sr, 'PCM_16')
            file_count += 1

        print("Resampled {0} wav files in sub-directory '{1}' to {2}Hz"
              .format(file_count, dir, new_sr))
        dir_count += 1
        total_file_count += file_count

    print("Complete!\nResampled a total of {0} wav files across {1} subdirectories"
          .format(total_file_count, dir_count))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--source_dir',
      type=str,
      default=None,
      help='Location of source wav files.')
  parser.add_argument(
      '--destination_dir',
      type=str,
      default=None,
      help='Where to save downsampled wav files.')
  parser.add_argument(
      '--resample_rate',
      type=int,
      default=None,
      help='The new desired sampling rate'
  )

  FLAGS, unparsed = parser.parse_known_args()
  downsample_and_save()