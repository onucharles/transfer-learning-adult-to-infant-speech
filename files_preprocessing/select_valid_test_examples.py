"""
Randomly select validation and test examples from dataset, and save file names to txt file.
Code assumes 2 classes.

"""

import os
import argparse

def main():
    # load positive and negative examples.

    # randomly select 10% for validation and 10% for testing.
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pos_samples_dir',
        type=str,
        default=None,
        help='Location of positive examples.')
    parser.add_argument(
        '--neg_samples_dir',
        type=str,
        default=None,
        help='Location of negative examples.')

    FLAGS, unparsed = parser.parse_known_args()
    main()