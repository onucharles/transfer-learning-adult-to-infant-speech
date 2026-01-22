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
from config.c_parameters import c_parameters


class ChillantoTimeshiftDataset(mod.SpeechDataset):
    """
    SpeechDataset class that supports adding of noise before processing raw audio into MFCC.
    """
    def __init__(self, data, set_type, config):
        super(ChillantoTimeshiftDataset, self).__init__(data, set_type, config)
        noise_samples = [librosa.core.load(file, sr=self.input_length) for file in config["bg_noise_files"]]
        self.bg_noise_audio = list([librosa.resample(sample, freq, config['sampling_freq'])
                                    for idx, (sample, freq) in enumerate(noise_samples)])
        self.crop = config["crop"]

    def preprocess(self, example, silence=False):
        in_len = self.input_length
        data = librosa.core.load(example, sr=self.sampling_freq)[0]
        data = np.pad(data, (0, max(0, in_len - len(data))), "constant")

        end_crop = int(np.floor(in_len * self.crop ))
        print('crop is {}. and endcrop is {}'.format(self.crop, end_crop))
        data = data[:in_len]
        data[0:end_crop] = 0
        print('max val is ', np.max(data))
        data = preprocess_audio(data, self.sampling_freq, self.n_mels, self.filters, self.frame_shift_ms, self.window_size_ms)

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
                        project_name="chillanto-timeshift-crop", workspace=os.getenv("COMET_WORKSPACE", "co-jl-transfer"))
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
        crop = int(config['crop'] * 100)
        experiment = config['experiment']
        experiment.log_metric("test_F1", f1, step=crop)
        experiment.log_metric("test_accuracy", avg_acc, step=crop)
        experiment.log_metric("test_precision", precision, step=crop)
        experiment.log_metric("test_recall", recall, step=crop)

        sens, spec, uar = evalutils.calc_sens_spec_uar(conf_mat, pos_class=3)
        experiment.log_metric('test_sensitivity', sens, step=crop)
        experiment.log_metric('test_specificity', spec, step=crop)
        experiment.log_metric('test_UAR', uar, step=crop)
        # experiment.log_metric("true_positives", tp, step=noise_pct)
        # experiment.log_metric("true_negatives", tn, step=noise_pct)
        # experiment.log_metric("false_positives", fp, step=noise_pct)
        # experiment.log_metric("false_negatives", fn, step=noise_pct)

def noisy_eval(pipeline, config, crops = [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]):
    """
    Evaluates the given model 'pipeline' on test data. adding certain amount & type of noise to the test data.
    """
    if not pipeline:
        raise("Passed SVM model is null.")

    for crop in crops:
        config['crop'] = crop
        # log 'crop' to comet
        if config['log_experiment']:
            experiment = config['experiment']
            experiment.log_metric('crop', crop)

        train_set, dev_set, test_set = ChillantoTimeshiftDataset.splits(config)
        print('Loaded {0} training\t{1} validation\t{2} test examples'.format(len(train_set), len(dev_set),
                                                                              len(test_set)))
        test_data, test_labels = ChillantoTimeshiftDataset.convert_dataset(test_set)

        y_pred_proba = pipeline.predict_proba(test_data)
        y_pred = pipeline.predict(test_data)
        print_metrics_and_log(test_labels, y_pred, config)
        break

        # test_data_loader = build_test_data_loader(config,ChillantoTimeshiftDataset)
        # evaluator = TimeshiftEvaluator(params)
        # evaluator.test_loader = test_data_loader
        # evaluator.step = int(crop * 100)
        # evaluator.evaluate()

def build_config():
    # config = {
    #     'seed': 5,
    #     'log_experiment': True,
    #     'input_file': '/mnt/hdd/Experiments/chillanto-svm/018e4b0f94654f508045373483b92a1f/train_eval.pkl',
    #     }


    # merge above config with dataset config.
    return dict(ChainMap(ChillantoTimeshiftDataset.default_config(), c_parameters))

def main():
    # load parameters
    config = build_config()

    # prepare experiment
    config = prepare_experiment(config)
    set_seed(config['seed'])

    print("Loading model from {0}...".format(config['source_model']))
    pipeline, _, _, _ = joblib.load(config['source_model'])
    noisy_eval(pipeline, config)

if __name__ == "__main__":
    main()
