"""
Trains classical machine learning models (e.g SVM, logistic regression).

"""
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

def prepare_experiment_directory(config):
    """ Sets up folders where all experiment files will be saved to.
    """
    # create unique sub-folder for this experiment
    exp_dir = config['output_folder'] + '/' + current_datetime() + '/'
    create_folder(exp_dir)
    print('Experiment files will be saved to: ', exp_dir)

    # Specify full path to model file.
    # config["model_file"] = exp_dir + "model.pt"

    # Specify logs directory
    config['exp_dir'] = exp_dir
    # config['logs_dir'] = exp_dir + 'logs/'
    # create_folder(config['logs_dir'])       # TODO remove this if using tensorboard to write logs.

    # save parameters to json file
    config_path = exp_dir + 'config.json'
    save_json(dict(config), config_path)
    print('Saved experiment parameters to: ', config_path)

def cross_validate(config):
    train_set, dev_set, test_set = mod.SpeechDataset.splits(config)
    print('Loaded {0} training\t{1} validation\t{2} test examples'.format(len(train_set), len(dev_set), len(test_set)))

    train_data, train_labels = mod.convert_dataset(train_set)
    valid_data, valid_labels = mod.convert_dataset(dev_set)
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
    train_data, train_labels = mod.convert_dataset(train_set)
    valid_data, valid_labels = mod.convert_dataset(dev_set)
    data = np.concatenate([train_data, valid_data])
    labels = np.concatenate([train_labels, valid_labels])
    print('Merged training and validation sets. Data dimension is {0} and labels {1}'.format(data.shape, labels.shape))

    # load test set
    test_data, test_labels = mod.convert_dataset(test_set)

    # make classifier and set hyperparams.
    clf_name = 'clf'
    C = config['svm_train_params']['C']
    gamma = config['svm_train_params']['gamma']
    clf = SVC(kernel='rbf', probability=True, cache_size=4000, C=C, gamma=gamma, class_weight='balanced')
    pipeline = Pipeline([('scaler', StandardScaler()), (clf_name, clf)])

    result = trainutils.train_eval(data, labels, test_data, test_labels, pipeline)

    # save results
    joblib.dump(result, os.path.join(config['exp_dir'], config['train_eval_file']))

    print("Experiment files saved to", config['exp_dir'])

def main():
    # load parameters
    builder = ConfigBuilder(
        c_parameters,
    )
    parser = builder.build_argparse()
    config = builder.config_from_argparse(parser)

    prepare_experiment_directory(config)

    # decide between model selection or training/eval
    if config['mode'] == 'model_selection':
        print('Running in model selection mode...')
        cross_validate(config)
    elif config['mode'] == 'train_eval':
        print('Running in train/eval mode...')
        train_eval(config)


if __name__ == "__main__":
    main()
