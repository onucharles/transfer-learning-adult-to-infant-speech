import os
import random
import re

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

class VCTKDataset(data.Dataset):
    def __init__(self, data, set_type, config):
        super().__init__()
        self.audio_files = list(data.keys())
        self.set_type = set_type
        self.audio_labels = list(data.values())
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
        data = librosa.core.load(example, sr=self.sampling_freq)[0]
        data = np.pad(data, (0, max(0, in_len - len(data))), "constant")
        start_indx = np.random.choice(np.arange(len(data) - self.input_length))
        data  = data[start_indx:][:self.input_length]
        if self.set_type == DS.TRAIN:
            data = self._timeshift_audio(data)

        data = torch.from_numpy(
            preprocess_audio(data, self.sampling_freq, self.n_mels, self.filters, self.frame_shift_ms, self.window_size_ms)
        )
        return data


    @classmethod
    def splits(cls, config):
        folder = config["data_folder"]

        sets = {
            DS.TRAIN : {},
            DS.DEV : {},
            DS.TEST : {}
        }

        added_speakers = []
        speakers = {}
        split_filename = 'iden_split.txt'
        with open(folder / split_filename) as f:
            file_splits = f.readlines()
        file_splits = [x.strip().split(' ') for x in file_splits]

        for (dset, fpath) in file_splits:
            speaker_id = fpath.split('/')[0]
            ds = DS.TRAIN
            if dset == '2':
                ds = DS.DEV
            if dset == '3':
                ds = DS.TEST

            add_speaker = True
            if config['label_limit'] != False:
                if len(added_speakers) + 1 > config['label_limit']:
                    add_speaker = False

            if add_speaker:
                if speaker_id not in added_speakers:
                    added_speakers.append(speaker_id)
                    speakers[speaker_id] = []
                full_path = folder / 'wav' / fpath
                sets[ds][full_path] = added_speakers.index(speaker_id)
                speakers[speaker_id].append(full_path)


        data_distribution = [ (k, len(files)) for (k, files) in speakers.items()]
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
        return self.preprocess(self.audio_files[index]), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels)


