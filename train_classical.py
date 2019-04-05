"""
Trains classical machine learning models (e.g SVM, logistic regression).

"""
from comet_ml import Experiment

import os
from ConfigBuilder import ConfigBuilder
import model as mod
from utils.ioutils import create_folder, current_datetime, save_json
from config.c_parameters import c_parameters
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from utils import trainutils
from sklearn.externals import joblib
import numpy as np
from utils import evalutils
from sklearn import metrics
import random


def prepare_experiment(config):
    """ Sets up folders where all experiment files will be saved to.
    """
    if not config['log_experiment']:
        return config

    experiment = Experiment(api_key="w7QuiECYXbNiOozveTpjc9uPg",
                        project_name="chillanto-svm", workspace="co-jl-transfer")
    experiment.add_tag('svm')
    exp_id = experiment.id

    # create unique sub-folder for this experiment
    exp_dir = config['output_folder'] + '/' + exp_id + '/'
    create_folder(exp_dir)
    print('Experiment files will be saved to: ', exp_dir)

    # Specify logs directory
    config['exp_dir'] = exp_dir

    # save parameters to json file
    config_path = exp_dir + 'config.json'
    save_json(dict(config), config_path)
    print('Saved experiment parameters to: ', config_path)

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
        experiment = config['experiment']
        experiment.log_metric("test_F1", f1)
        experiment.log_metric("test_accuracy", avg_acc)
        experiment.log_metric("test_precision", precision)
        experiment.log_metric("test_recall", recall)
        experiment.log_metric("true_positives", tp)
        experiment.log_metric("true_negatives", tn)
        experiment.log_metric("false_positives", fp)
        experiment.log_metric("false_negatives", fn)

        sens, spec, uar = evalutils.calc_sens_spec_uar(conf_mat, pos_class=3)
        experiment.log_metric('test_sensitivity', sens)
        experiment.log_metric('test_specificity', spec)
        experiment.log_metric('test_UAR', uar)

def cross_validate(config):
    train_set, dev_set, test_set = mod.SpeechDataset.splits(config)
    print('Loaded {0} training\t{1} validation\t{2} test examples'.format(len(train_set), len(dev_set), len(test_set)))

    train_data, train_labels = mod.SpeechDataset.convert_dataset(train_set)
    valid_data, valid_labels = mod.SpeechDataset.convert_dataset(dev_set)
    data = np.concatenate([train_data, valid_data])
    labels = np.concatenate([train_labels, valid_labels])
    print('Merged training and validation sets. Data dimension is {0} and labels {1}'.format(data.shape, labels.shape))

    # make pipeline and set hyperparameters ranges to be searched.
    clf_name = 'clf'
    clf = SVC(kernel='rbf', probability=True, cache_size=4000, class_weight='balanced')
    pipeline = Pipeline([('scaler', StandardScaler()), (clf_name, clf)])
    hyperparams = trainutils.make_hyperparams(clf_name, config['svm_hyperparam_range'])

    # run grid search.
    grid_search = trainutils.model_select_skfold(data, labels, pipeline, hyperparams, n_splits=config['svm_no_folds'])

    # save results
    joblib.dump(grid_search, os.path.join(config['exp_dir'], config['grid_search_file']))

    print("Experiment files saved to", config['exp_dir'])

def train_eval(config):
    train_set, dev_set, test_set = mod.SpeechDataset.splits(config)
    print('Loaded {0} training\t{1} validation\t{2} test examples'.format(len(train_set), len(dev_set), len(test_set)))

    # merge train and validation sets into one.
    train_data, train_labels = mod.SpeechDataset.convert_dataset(train_set)
    valid_data, valid_labels = mod.SpeechDataset.convert_dataset(dev_set)
    data = np.concatenate([train_data, valid_data])
    labels = np.concatenate([train_labels, valid_labels])
    print('Merged training and validation sets. Data dimension is {0} and labels {1}'.format(data.shape, labels.shape))

    # load test set
    test_data, test_labels = mod.SpeechDataset.convert_dataset(test_set)

    # make classifier and set hyperparams.
    clf_name = 'clf'
    C = config['svm_train_params']['C']
    gamma = config['svm_train_params']['gamma']
    clf = SVC(kernel='rbf', probability=True, cache_size=4000, C=C, gamma=gamma, class_weight='balanced')
    pipeline = Pipeline([('scaler', StandardScaler()), (clf_name, clf)])

    result = trainutils.train_eval(data, labels, test_data, test_labels, pipeline)
    pipeline, y_test, y_pred, y_pred_proba = result
    print_metrics_and_log(y_test, y_pred, config)

    # save results
    if config['log_experiment']:
        # save to file
        result_file_path = os.path.join(config['exp_dir'], config['train_eval_file'])
        joblib.dump(result, result_file_path)
        print("Experiment files saved to", config['exp_dir'])

        # log to comet
        experiment = config['experiment']
        experiment.log_asset(result_file_path, overwrite=True)

def eval(pipeline, config):
    """
    Evaluates the given model 'pipeline' on test data.
    """
    if not pipeline:
        raise("Passed SVM model is null.")

    train_set, dev_set, test_set = mod.SpeechDataset.splits(config)
    print('Loaded {0} training\t{1} validation\t{2} test examples'.format(len(train_set), len(dev_set), len(test_set)))
    test_data, test_labels = mod.SpeechDataset.convert_dataset(test_set)

    y_pred_proba = pipeline.predict_proba(test_data)
    y_pred = pipeline.predict(test_data)
    print_metrics_and_log(test_labels, y_pred, config)


def main():
    # load parameters
    builder = ConfigBuilder(
        c_parameters,
    )
    parser = builder.build_argparse()
    parser.add_argument("--seed", type=int)
    config = builder.config_from_argparse(parser)
    print('seed is ', config['seed'])

    # prepare experiment
    config = prepare_experiment(config)

    set_seed(config['seed'])

    # decide between model selection or training/eval
    if config['mode'] == 'model_selection':
        print('Running in model selection mode...')
        cross_validate(config)
    elif config['mode'] == 'train_eval':
        print('Running in train/eval mode...')
        train_eval(config)
    elif config['mode'] == 'eval':
        print('Running in eval mode...')
        pipeline, _,_,_ = joblib.load(config['source_model'])
        eval(pipeline, config)
    else:
        raise("Unknown mode specified.")

if __name__ == "__main__":
    main()
