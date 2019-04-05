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

from .dataset_type import DatasetType as DS



class ESC50Dataset(data.Dataset):
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
        config["input_length"] = 8000
        config["timeshift_ms"] = 100
        config["train_pct"] = 60
        config["dev_pct"] = 20
        config["test_pct"] = 20
        config["sampling_freq"] = 8000
        config["n_dct_filters"] = 40
        config["n_mels"] = 40
        # add Unknown and Silence
        config["window_size_ms"] = 30
        config["frame_shift_ms"] = 10
        config["label_limit"] = False

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
        if len(data) - self.input_length == 0:
            start_indx = 0
        else:
            start_indx = np.random.choice(np.arange(len(data) - self.input_length))
        data  = data[start_indx:][:self.input_length]
        if self.set_type == DS.TRAIN:
            data = self._timeshift_audio(data)

        data = torch.from_numpy(
            preprocess_audio(data, self.sampling_freq, self.n_mels, self.filters, self.frame_shift_ms, self.window_size_ms)
        )
        return data


    @classmethod
    def rand_add_category_wavs(cls, existing, files, num, label):
        files_to_add = set(np.random.choice(list(files), num, replace=False))
        for wav in files_to_add:
            existing[wav] = label
        return existing

    @classmethod
    def splits(cls, config):
        folder = config["data_folder"]
        train_pct = config["train_pct"]
        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]

        sets = {
            DS.TRAIN : {},
            DS.DEV : {},
            DS.TEST : {}
        }


        ESC_10_CATEGORIES = (0, 1, 10, 11, 12, 20, 21, 38, 40, 41)
        all_categories = {}
        all_category_ids = []
        for path in glob.iglob(f"{folder}/*.wav"):
            path_parts = path.split('-')
            category_id = int(path_parts[-1].split('.wav')[0])
            if category_id not in ESC_10_CATEGORIES:
                continue

            if category_id not in all_categories:
                all_categories[category_id] = []
                all_category_ids.append(category_id)

            all_categories[category_id].append(path)

        categories = {}
        if config['label_limit'] == False:
            categories = all_categories
        else:
            for key in np.random.choice(all_category_ids, config['label_limit'], replace=False):
                categories[key] = all_categories[key]

        data_distribution = [ (k, len(files)) for (k, files) in categories.items()]
        total = np.sum([count for (_, count) in data_distribution])
        print(f"Total files: {total}")
        print(f"Total labels: {len(data_distribution)}")

        labels = {label: i for i, (label, _) in enumerate(data_distribution)}

        for category_id, count in data_distribution:
            train_num = int(np.floor(count * train_pct / 100 ))
            dev_num = int(np.floor(count * dev_pct / 100 ))
            test_num = int(np.floor(count * test_pct / 100))

            files = set(categories[category_id])
            sets[DS.TRAIN] = cls.rand_add_category_wavs(sets[DS.TRAIN], files, train_num, labels[category_id])

            files = files - set(sets[DS.TRAIN].keys())
            sets[DS.DEV] = cls.rand_add_category_wavs(sets[DS.DEV], files, dev_num, labels[category_id])

            files = files - set(sets[DS.DEV].keys())
            sets[DS.TEST] = cls.rand_add_category_wavs(sets[DS.TEST], files, test_num, labels[category_id])

        bg_noise_files = []
        print("labels are: ", labels)
        train_cfg = ChainMap(dict(bg_noise_files=bg_noise_files), config)
        test_cfg = ChainMap(dict(bg_noise_files=bg_noise_files, noise_prob=0), config)
        datasets = (cls(sets[DS.TRAIN], DS.TRAIN, train_cfg),
                cls(sets[DS.DEV], DS.DEV, test_cfg),
                cls(sets[DS.TEST], DS.TEST, test_cfg))
        return {
                'datasets': datasets,
                # TODO: there is a strong assumption that this is sorted:
                'distribution': { lbl: count for lbl, count in data_distribution },
                'labels': labels,
                'n_labels': len(labels),
                'total': total
                }

    @staticmethod
    def convert_dataset(data_set):
        """
        Converts a model.SpeechDataset object into a plain data array and label vector
        :param data_set:
        :return:
        """
        data = []
        labels = []
        for i in range(len(data_set)):
            mfcc_img, class_label = data_set[i]
            data.append(mfcc_img.numpy().flatten())
            labels.append(class_label)

        data = np.array(data)
        labels = np.array(labels)
        print('data converted to array of: {0}. and labels: {1}'.format(data.shape, labels.shape))
        return data, labels

    def __getitem__(self, index):
        return self.preprocess(self.audio_files[index]), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels)

