"""
Experiment for evaluating saved SVM models on chillanto test set, where noise is added to test data.
"""

from comet_ml import Experiment

import os
from ConfigBuilder import ConfigBuilder
import model as mod
from utils.ioutils import create_folder, current_datetime, save_json
from sklearn.externals import joblib
import numpy as np
from utils import evalutils
from sklearn import metrics
import random
import torch
import librosa
from manage_audio import preprocess_audio
from collections import ChainMap
from pathlib import Path


class ChillantoNoiseMixDataset(mod.SpeechDataset):
    """
    SpeechDataset class that supports adding of noise before processing raw audio into MFCC.
    """
    def __init__(self, data, set_type, config):
        super(ChillantoNoiseMixDataset, self).__init__(data, set_type, config)
        self.audio_files = list(data.keys())
        self.set_type = set_type
        self.audio_labels = list(data.values())
        self.input_length = config["input_length"]
        sample, freq = librosa.load(config['noise_files'][0], sr=None)
        self.bg_noise_audio = librosa.resample(sample, freq, config['sampling_freq'])
        print(np.shape(self.bg_noise_audio))
        self.unknown_prob = config["unknown_prob"]
        self.silence_prob = config["silence_prob"]
        self.noise_prob = config["noise_prob"]
        self.n_dct = config["n_dct_filters"]
        self.input_length = config["input_length"]
        self.timeshift_ms = config["timeshift_ms"]
        self.filters = librosa.filters.dct(config["n_dct_filters"], config["n_mels"])
        self.n_mels = config["n_mels"]
        n_unk = len(list(filter(lambda x: x == 1, self.audio_labels)))
        self.n_silence = int(self.silence_prob * (len(self.audio_labels) - n_unk))
        self.sampling_freq = config["sampling_freq"]
        self.window_size_ms = config["window_size_ms"]
        self.frame_shift_ms = config["frame_shift_ms"]
        self.noise_pct = config['noise_pct']
        self.noise_type = config['noise_type']

        print('noise files are: ', config['noise_files'])

    def preprocess(self, example, silence=False):
        in_len = self.input_length
        data = librosa.core.load(example, sr=self.sampling_freq)[0]
        data = np.pad(data, (0, max(0, in_len - len(data))), "constant")

        bg_noise = self.bg_noise_audio
        if self.noise_type == 'gaussian':
            noise_sample = np.random.normal(0, 0.1, in_len)
        else:
            noise_sample = bg_noise[:in_len]

        # mix the noise into the data:
        noise = self.noise_pct * noise_sample
        data = noise + data[:in_len]

        data = torch.from_numpy(
            preprocess_audio(data, self.sampling_freq, self.n_mels, self.filters, self.frame_shift_ms, self.window_size_ms)
        )
        return data


    @staticmethod
    def default_config():
        config = {}
        config["group_speakers_by_id"] = True
        config["silence_prob"] = 0.0
        config["noise_prob"] = 0.0
        config["input_length"] = 8000
        config["timeshift_ms"] = 100
        config["unknown_prob"] = 0.0
        config["train_pct"] = 80
        config["dev_pct"] = 5
        config["test_pct"] = 40
        config["wanted_words"] = ["normal", "asphyxia"]
        config["data_folder"] = "/mnt/hdd/Datasets/chillanto-8k-16bit-renamed"
        config["sampling_freq"] = 8000
        config["n_dct_filters"] = 40
        config["n_mels"] = 40
        config["window_size_ms"] = 30
        config["frame_shift_ms"] = 10
        config["cache_size"] = 32768
        return config

def set_noise_files(noise_type):
    if noise_type == 'gaussian':
        return [ Path('/mnt/hdd/Datasets/noise') / 'gaussian_0_1_noise.wav' ]
    elif noise_type == 'dog_bark':
        return [ Path('/mnt/hdd/Datasets/noise') / 'dog_bark' / f'{fn}.wav' for fn in [4, 15, 68, 71, 97, 160, 163, 164]]
    elif noise_type == 'children_playing':
        return [ Path('/mnt/hdd/Datasets/noise') / 'children_playing' / f'{fn}.wav' for fn in [6, 32, 44, 54, 56, 67, 87, 134, 152, 174]]
    elif noise_type == 'siren':
        return [ Path('/mnt/hdd/Datasets/noise') / 'siren' / f'{fn}.wav' for fn in [0, 3, 18, 27, 36, 43, 50, 60, 90, 92]]

def prepare_experiment(config):
    """ Sets up folders where all experiment files will be saved to.
    """
    if not config['log_experiment']:
        return config

    experiment = Experiment(api_key="w7QuiECYXbNiOozveTpjc9uPg",
                        project_name="chillanto-noise", workspace="co-jl-transfer")
    experiment.add_tag('svm')
    # exp_id = experiment.id

    # # create unique sub-folder for this experiment
    # exp_dir = config['output_folder'] + '/' + exp_id + '/'
    # create_folder(exp_dir)
    # print('Experiment files will be saved to: ', exp_dir)
    #
    # # Specify logs directory
    # config['exp_dir'] = exp_dir
    #
    # # save parameters to json file
    # config_path = exp_dir + 'config.json'
    # save_json(dict(config), config_path)
    # print('Saved experiment parameters to: ', config_path)

    experiment.log_parameters(dict(config))
    config['experiment'] = experiment

    return config

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def print_metrics_and_log(y_true, y_pred, config):
    """
    Print evaluation metrics and log to comet
    """
    # compute and print metrics. acc, precision, recall, specificity, f1
    conf_mat = metrics.confusion_matrix(y_true, y_pred, labels=np.arange(4))  # TODO remove this hard number '4'
    tp, tn, fp, fn, p, n = evalutils.read_conf_matrix(conf_mat, pos_class=3)
    f1, precision, recall = evalutils.f1_prec_recall(tp, tn, fp, fn, p, n)
    avg_acc = (tp + tn) / (tp + tn + fp + fn)
    print("Confusion matrix: {}".format(conf_mat))
    print("{} accuracy: {}\tF1 = {}\tPrecision={}\tRecall={}".format('SVM', avg_acc, f1, precision, recall))

    # log to comet
    if config['log_experiment']:
        noise_pct = config['noise_pct'] * 100
        experiment = config['experiment']
        experiment.log_metric("test_F1", f1, step=noise_pct)
        experiment.log_metric("test_accuracy", avg_acc, step=noise_pct)
        experiment.log_metric("test_precision", precision, step=noise_pct)
        experiment.log_metric("test_recall", recall, step=noise_pct)

        sens, spec, uar = evalutils.calc_sens_spec_uar(conf_mat, pos_class=3)
        experiment.log_metric('test_sensitivity', sens, step=noise_pct)
        experiment.log_metric('test_specificity', spec, step=noise_pct)
        experiment.log_metric('test_UAR', uar, step=noise_pct)
        # experiment.log_metric("true_positives", tp, step=noise_pct)
        # experiment.log_metric("true_negatives", tn, step=noise_pct)
        # experiment.log_metric("false_positives", fp, step=noise_pct)
        # experiment.log_metric("false_negatives", fn, step=noise_pct)

def noisy_eval(pipeline, config, noise_range=[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    """
    Evaluates the given model 'pipeline' on test data. adding certain amount & type of noise to the test data.
    """
    if not pipeline:
        raise("Passed SVM model is null.")

    config['noise_files'] = set_noise_files(config['noise_type'])

    for pct in noise_range:
        config['noise_pct'] = float(pct)

        # log 'noise_pct' to comet
        if config['log_experiment']:
            experiment = config['experiment']
            experiment.log_metric('noise_pct', pct)

        train_set, dev_set, test_set = ChillantoNoiseMixDataset.splits(config)
        print('Loaded {0} training\t{1} validation\t{2} test examples'.format(len(train_set), len(dev_set), len(test_set)))
        test_data, test_labels = ChillantoNoiseMixDataset.convert_dataset(test_set)

        y_pred_proba = pipeline.predict_proba(test_data)
        y_pred = pipeline.predict(test_data)
        print_metrics_and_log(test_labels, y_pred, config)

def build_config():
    config = {
        'seed': 10,
        #'output_folder': '/mnt/hdd/Experiments/chillanto-noise',
        'log_experiment': True,
        #'noise_type': 'children_playing',
        #'input_file': '/mnt/hdd/Experiments/chillanto-svm/386a7ec3b7524234afad3e28864dbecf/train_eval.pkl',
        #'input_file': '/mnt/hdd/Experiments/chillanto-svm/65ac48c462644ae08f1f8dd2c2d1fb1b/train_eval.pkl',
        #'input_file': '/mnt/hdd/Experiments/chillanto-svm/0313b6c829cf41b489af0c76423ab813/train_eval.pkl',
        #'input_file': '/mnt/hdd/Experiments/chillanto-svm/78716497b70449a5944596777e467d06/train_eval.pkl',
        #'input_file': '/mnt/hdd/Experiments/chillanto-svm/b43dc83c36024ceca379fb75385b205e/train_eval.pkl',
        }

    # merge above config with dataset config.
    return dict(ChainMap(ChillantoNoiseMixDataset.default_config(), config))

def main():
    # load parameters
    config = build_config()

    builder = ConfigBuilder(config)
    parser = builder.build_argparse()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--noise_type", type=str)
    config = builder.config_from_argparse(parser)

    # prepare experiment
    config = prepare_experiment(config)
    set_seed(config['seed'])

    print("Loading model from {0}...".format(config['input_file']))
    pipeline, _, _, _ = joblib.load(config['input_file'])
    noisy_eval(pipeline, config)

if __name__ == "__main__":
    main()
