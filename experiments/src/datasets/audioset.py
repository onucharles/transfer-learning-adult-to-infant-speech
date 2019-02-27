import hashlib
import math
import os
import random
import re
import h5py

from collections import ChainMap, Counter

from torch.autograd import Variable
import librosa
import numpy as np
import glob
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from .manage_audio import preprocess_audio
from .simple_cache import SimpleCache

from .dataset_type import DatasetType as DS

class AudioSetDataset(data.Dataset):
    def __init__(self, data, set_type, config):
        super().__init__()
        self.audio_files = list([ d for (d, _) in data])
        self.set_type = set_type
        self.audio_labels = list([ l for (_, l) in data])
        self.input_length = config["input_length"]
        self.n_dct = config["n_dct_filters"]
        self.input_length = config["input_length"]
        self.timeshift_ms = config["timeshift_ms"]
        self.filters = librosa.filters.dct(config["n_dct_filters"], config["n_mels"])
        self.n_mels = config["n_mels"]
        self.sampling_freq = config["sampling_freq"]
        self.window_size_ms = config["window_size_ms"]
        self.frame_shift_ms = config["frame_shift_ms"]

    @staticmethod
    def default_config(custom_config={}):
        """ NOTE: you must provide a `data_folder` """
        config = {}
        config["group_speakers_by_id"] = True
        config["input_length"] = 8000
        config["timeshift_ms"] = 100
        config["sampling_freq"] = 16000
        config["n_dct_filters"] = 40
        config["n_mels"] = 40
        # add Unknown and Silence
        config["window_size_ms"] = 25
        config["frame_shift_ms"] = 10
        config["label_limit"] = False
        config["loss"] = "hinge"

        return dict(ChainMap(custom_config, config))

    def _timeshift_audio(self, data):
        shift = (self.sampling_freq * self.timeshift_ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(data, (a, b), "constant")
        return data[:len(data) - a] if a else data[b:]

    def preprocess(self, example, silence=False):
        in_len = self.input_length
        example = (np.float32(example) - 128.) / 128.
        data = torch.from_numpy(example)
        return data


    @classmethod
    def load_data(cls, hdf5_path):
        with h5py.File(hdf5_path, 'r') as hf:
            x = hf['x'][:]
            y = hf['y'][:]
            video_id_list = hf['video_id_list'][:].tolist()

        return x, y, video_id_list



    @classmethod
    def splits(cls, config):
        folder = config["data_folder"]
        train = folder / 'bal_train.h5'
        test = folder / 'eval.h5'
        (xs, ys, vlist)  = cls.load_data(train)
        (xs_test, ys_test, _)  = cls.load_data(test)

        n_dev = int(np.floor(len(xs) * 0.2))
        all_indexes = np.arange(len(xs))
        dev_indexes = set(np.random.choice(all_indexes, n_dev, replace=False))
        train_indexes = set(all_indexes) - dev_indexes
        assert len(train_indexes) + len(dev_indexes) == len(xs)

        sets = {
            DS.TRAIN : [ (xs[idx], np.argwhere(ys[idx] == True)[0][0]) for idx in train_indexes],
            DS.DEV : [ (xs[idx], np.argwhere(ys[idx] == True)[0][0]) for idx in train_indexes],
            DS.TEST : [ (xs_test[idx], np.argwhere(ys_test[idx] == True)[0][0]) for idx in np.arange(len(xs_test))]
        }

        data_distribution = Counter([ lbl for (data, lbl) in sets[DS.TRAIN] ]).items()
        total = np.sum([count for (_, count) in data_distribution])
        labels = {label: i for i, (label, _) in enumerate(data_distribution)}
        print(f"Total files: {total}")
        print(f"Distribution: {data_distribution}")
        print("labels are: ", labels)
        datasets = (cls(sets[DS.TRAIN], DS.TRAIN, config), cls(sets[DS.DEV], DS.DEV, config), cls(sets[DS.TEST], DS.TEST, config))
        return {
                'datasets': datasets,
                'distribution': { lbl: count for lbl, count in data_distribution },
                'labels': labels,
                'n_labels': len(labels),
                'total': total
                }

    def __getitem__(self, index):
        print(self.audio_labels[index])
        return self.preprocess(self.audio_files[index]), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels)


