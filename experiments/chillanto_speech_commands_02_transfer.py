from pathlib import Path
from collections import ChainMap

from src.settings import CHILLANTO_DATA_FOLDER, CHILLANTO_LOGGING_FOLDER, CHILLANTO_MODELS_FOLDER
from src.datasets.chillanto import ChillantoDataset, chillanto_sampler
from src.tasks.train_and_evaluate import task_train_and_evaluate, task_config, setup_task, build_data_loaders
from src.training_helpers import load_weights

params_to_load = ['conv0.weight', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked',
                       'conv1.weight', 'bn2.running_mean', 'bn2.running_var', 'bn2.num_batches_tracked',
                       'conv2.weight', 'bn3.running_mean', 'bn3.running_var', 'bn3.num_batches_tracked',
                       'conv3.weight', 'bn4.running_mean', 'bn4.running_var', 'bn4.num_batches_tracked',
                       'conv4.weight', 'bn5.running_mean', 'bn5.running_var', 'bn5.num_batches_tracked',
                       'conv5.weight', 'bn6.running_mean', 'bn6.running_var', 'bn6.num_batches_tracked',
                       'conv6.weight']


def build_config():
    config = task_config({
            'project': 'chillanto_speech_commands_transfer',
            'model_path': CHILLANTO_MODELS_FOLDER / 'trans_chill_sc',
            'log_file_path': CHILLANTO_LOGGING_FOLDER /  'logs.pkl',
            'predictions_path': CHILLANTO_LOGGING_FOLDER / 'predictions.pkl',
            "data_folder": CHILLANTO_DATA_FOLDER,
            'print_confusion_matrix': True,
            'lr': [0.001, 0.0001],
            'weight_decay': 0.00001,
            'momentum': 0.9,
            'schedule': [400],
            'n_epochs': 30,
            'n_labels': 4,
            'silence_prob': 0.0,
            'noise_prob': 0.0,
            'unknown_prob': 0.0,
            'input_length': 8000,
            'wanted_words': ['normal8k', 'asphyxia8k'],
            'batch_size': 50,
            'cache_size': 0,
            "dev_pct": 5,
            "test_pct": 40,
            "sampling_freq": 8000,
            'cache_size':32768,
            })

    # Merge together the model, training and dataset configuration:
    return dict(ChainMap(ChillantoDataset.default_config(config), config))


def model_transfer(source_model_path):
    config = build_config()
    data_loaders = build_data_loaders(config, ChillantoDataset, chillanto_sampler)
    params = setup_task(config, data_loaders, 4)
    load_weights(params['model'], source_model_path, params_to_load)
    task_train_and_evaluate(params)
