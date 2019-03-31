from enum import Enum
import hashlib
import math
import os
import random
import re

from collections import ChainMap

from torch.autograd import Variable
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from .manage_audio import preprocess_audio
from .simple_cache import SimpleCache
from .dataset_type import DatasetType


def chillanto_sampler(train_set, config):
    # TODO fix this turkey
    class_prob = [0, 0, 0.76, 0.24]
    sample_weights = []

    for i in range(len(train_set)):
        _, label = train_set[i]
        sample_weights.append(1 / class_prob[label])

    sample_weights = torch.Tensor(sample_weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(train_set))

    return sampler

class ChillantoNoiseMixDataset(data.Dataset):
    LABEL_SILENCE = "__silence__"
    LABEL_UNKNOWN = "__unknown__"
    def __init__(self, data, set_type, config):
        super().__init__()
        self.audio_files = list(data.keys())
        self.set_type = set_type
        self.audio_labels = list(data.values())
        self.input_length = config["input_length"]
        config["bg_noise_files"] = list(filter(lambda x: str(x).endswith("wav"), config.get("bg_noise_files", [])))
        noise_samples = [librosa.core.load(str(file), sr=self.input_length) for file in config["bg_noise_files"]]

        #noise_samples = [librosa.load(file) for file in ['/network/data1/maloneyj/noise/siren/60.wav']]
        self.bg_noise_audio = list([librosa.resample(sample, freq, config['sampling_freq']) for idx, (sample, freq) in enumerate(noise_samples)])
        self.unknown_prob = config["unknown_prob"]
        self.silence_prob = config["silence_prob"]
        self.noise_prob = config["noise_prob"]
        self.n_dct = config["n_dct_filters"]
        self.input_length = config["input_length"]
        self.timeshift_ms = config["timeshift_ms"]
        self.filters = librosa.filters.dct(config["n_dct_filters"], config["n_mels"])
        self.n_mels = config["n_mels"]
        self._audio_cache = SimpleCache(config["cache_size"])
        self._file_cache = SimpleCache(config["cache_size"])
        n_unk = len(list(filter(lambda x: x == 1, self.audio_labels)))
        self.n_silence = int(self.silence_prob * (len(self.audio_labels) - n_unk))
        self.sampling_freq = config["sampling_freq"]
        self.window_size_ms = config["window_size_ms"]
        self.frame_shift_ms = config["frame_shift_ms"]
        self.noise_pct = config['noise_pct']

    @staticmethod
    def default_config(custom):
        config = {}
        config["group_speakers_by_id"] = True
        config["silence_prob"] = 0.
        config["noise_prob"] = 0.
        config["input_length"] = 8000
        config["timeshift_ms"] = 100
        config["cache_size"] =32768
        config["seed"] = 11
        config["unknown_prob"] = 0.
        config["train_pct"] = 80
        config["dev_pct"] = 10
        config["test_pct"] = 10
        config["wanted_words"] = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
        config["data_folder"] = ""#""/mnt/hdd/Datasets/speech-commands-8k-16bit"
        config["sampling_freq"] = 8000
        config["n_dct_filters"] = 40
        config["n_mels"] = 40
        config["n_feature_maps"] = 45
        config["window_size_ms"] = 30
        config["frame_shift_ms"] = 10
        config["noise_pct"] = 0.
        config["loss"] = "hinge"
        return ChainMap(custom,config)

    def preprocess(self, example, silence=False):
        in_len = self.input_length
        data = librosa.core.load(example, sr=self.sampling_freq)[0]
        data = np.pad(data, (0, max(0, in_len - len(data))), "constant")

        #bg_noise = np.random.choice(self.bg_noise_audio)
        bg_noise = self.bg_noise_audio[0]
        bg_noise = np.pad(bg_noise, (0, max(0, in_len - len(bg_noise))), "constant")
        noise_sample = bg_noise[:in_len]
        #noise_range = random.randint(0, len(bg_noise) - in_len - 1)
        #noise_sample = bg_noise[noise_range:noise_range + in_len]
        # mix the noise into the data:
        data = self.noise_pct * noise_sample[:in_len] + data[:in_len]

        data = torch.from_numpy(
            preprocess_audio(data, self.sampling_freq, self.n_mels, self.filters, self.frame_shift_ms, self.window_size_ms)
        )
        return data

    @classmethod
    def splits(cls, config):
        folder = config["data_folder"]
        wanted_words = config["wanted_words"]
        unknown_prob = config["unknown_prob"]
        train_pct = config["train_pct"]
        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]

        words = {word: i + 2 for i, word in enumerate(wanted_words)}
        words.update({cls.LABEL_SILENCE:0, cls.LABEL_UNKNOWN:1})
        sets = [{}, {}, {}]
        unknowns = [0] * 3
        bg_noise_files = []
        unknown_files = []

        for folder_name in os.listdir(folder):
            path_name = os.path.join(folder, folder_name)
            is_bg_noise = False
            if os.path.isfile(path_name):
                continue
            if folder_name in words:
                label = words[folder_name]
            elif folder_name == "_background_noise_":
                is_bg_noise = True
            else:
                label = words[cls.LABEL_UNKNOWN]

            for filename in os.listdir(path_name):
                wav_name = os.path.join(path_name, filename)
                if is_bg_noise and os.path.isfile(wav_name):
                    bg_noise_files.append(wav_name)
                    continue
                elif label == words[cls.LABEL_UNKNOWN]:
                    unknown_files.append(wav_name)
                    continue
                if config["group_speakers_by_id"]:
                    hashname = re.sub(r"_nohash_.*$", "", filename)
                max_no_wavs = 2**27 - 1
                bucket = int(hashlib.sha1(hashname.encode()).hexdigest(), 16)
                bucket = (bucket % (max_no_wavs + 1)) * (100. / max_no_wavs)
                if bucket < dev_pct:
                    tag = DatasetType.DEV
                elif bucket < test_pct + dev_pct:
                    tag = DatasetType.TEST
                else:
                    tag = DatasetType.TRAIN
                sets[tag.value][wav_name] = label

        for tag in range(len(sets)):
            unknowns[tag] = int(unknown_prob * len(sets[tag]))
        random.shuffle(unknown_files)
        a = 0
        for i, dataset in enumerate(sets):
            b = a + unknowns[i]
            unk_dict = {u: words[cls.LABEL_UNKNOWN] for u in unknown_files[a:b]}
            dataset.update(unk_dict)
            a = b

        print("labels are: ", words)
        #train_cfg = ChainMap(dict(bg_noise_files=bg_noise_files), config)
        #test_cfg = ChainMap(dict(bg_noise_files=bg_noise_files, noise_prob=0), config)
        datasets = (cls(sets[0], DatasetType.TRAIN, config), cls(sets[1], DatasetType.DEV, config), cls(sets[2], DatasetType.TEST, config))
        return datasets

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
        if index >= len(self.audio_labels):
            return self.preprocess(None, silence=True), 0
        return self.preprocess(self.audio_files[index]), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels) + self.n_silence
