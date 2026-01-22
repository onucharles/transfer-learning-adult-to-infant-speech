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


class ChillantoFreqMaskDataset(mod.SpeechDataset):
    """
    SpeechDataset class that supports adding of noise before processing raw audio into MFCC.
    """
    def __init__(self, data, set_type, config):
        super(ChillantoFreqMaskDataset, self).__init__(data, set_type, config)
        noise_samples = [librosa.core.load(file, sr=self.input_length) for file in config["bg_noise_files"]]
        self.bg_noise_audio = list([librosa.resample(sample, freq, config['sampling_freq'])
                                    for idx, (sample, freq) in enumerate(noise_samples)])
        self.freq_range = config["freq_range"]

    def preprocess(self, example, silence=False):
        in_len = self.input_length
        data = librosa.core.load(example, sr=self.sampling_freq)[0]
        data = np.pad(data, (0, max(0, in_len - len(data))), "constant")

        data = data[:in_len]

        data = preprocess_audio(data, self.sampling_freq, self.n_mels, self.filters, self.frame_shift_ms, self.window_size_ms)
        #data[freq_start:freq_end] = masked[freq_start:freq_end]
        # block out MEL for the whole time:
        data[:,self.freq_range] = np.zeros(101)
        data = torch.from_numpy(data)
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

def prepare_experiment(config):
    """ Sets up folders where all experiment files will be saved to.
    """
    if not config['log_experiment']:
        return config

    experiment = Experiment(api_key=os.getenv("COMET_API_KEY"),
                        project_name="chillanto-frequency-mask", workspace=os.getenv("COMET_WORKSPACE", "co-jl-transfer"))
    experiment.add_tag('svm')

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
        freq_range = config['freq_range'] + 1
        experiment = config['experiment']
        experiment.log_metric("test_F1", f1, step=freq_range)
        experiment.log_metric("test_accuracy", avg_acc, step=freq_range)
        experiment.log_metric("test_precision", precision, step=freq_range)
        experiment.log_metric("test_recall", recall, step=freq_range)

        sens, spec, uar = evalutils.calc_sens_spec_uar(conf_mat, pos_class=3)
        experiment.log_metric('test_sensitivity', sens, step=freq_range)
        experiment.log_metric('test_specificity', spec, step=freq_range)
        experiment.log_metric('test_UAR', uar, step=freq_range)
        # experiment.log_metric("true_positives", tp, step=noise_pct)
        # experiment.log_metric("true_negatives", tn, step=noise_pct)
        # experiment.log_metric("false_positives", fp, step=noise_pct)
        # experiment.log_metric("false_negatives", fn, step=noise_pct)

def noisy_eval(pipeline, config):
    """
    Evaluates the given model 'pipeline' on test data. adding certain amount & type of noise to the test data.
    """
    if not pipeline:
        raise("Passed SVM model is null.")

    for freq_range in range(40):
        config['freq_range'] = freq_range
        # log 'crop' to comet
        if config['log_experiment']:
            experiment = config['experiment']
            experiment.log_metric('freq_range', str(freq_range))

        train_set, dev_set, test_set = ChillantoFreqMaskDataset.splits(config)
        print('Loaded {0} training\t{1} validation\t{2} test examples'.format(len(train_set), len(dev_set),
                                                                              len(test_set)))
        test_data, test_labels = ChillantoFreqMaskDataset.convert_dataset(test_set)

        y_pred_proba = pipeline.predict_proba(test_data)
        y_pred = pipeline.predict(test_data)
        print_metrics_and_log(test_labels, y_pred, config)

def build_config():
    config = {
        'seed': 10,
        #'output_folder': '/mnt/hdd/Experiments/chillanto-noise',
        'log_experiment': True,
        'input_file': '/mnt/hdd/Experiments/chillanto-svm/60f7804db83841068b559624bd4ac899/train_eval.pkl',
        }

    # merge above config with dataset config.
    return dict(ChainMap(ChillantoFreqMaskDataset.default_config(), config))

def main():
    # load parameters
    config = build_config()

    # prepare experiment
    config = prepare_experiment(config)
    set_seed(config['seed'])

    print("Loading model from {0}...".format(config['input_file']))
    pipeline, _, _, _ = joblib.load(config['input_file'])
    noisy_eval(pipeline, config)

if __name__ == "__main__":
    main()