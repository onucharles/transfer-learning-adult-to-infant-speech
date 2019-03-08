from pathlib import Path
from collections import ChainMap
from src.settings import CHILLANTO_DATA_FOLDER, CHILLANTO_LOGGING_FOLDER, CHILLANTO_MODELS_FOLDER, CHILLANTO_NOISE_DATA_FOLDER
from src.datasets.chillanto_noise_mix import ChillantoNoiseMixDataset, chillanto_sampler
from src.training_helpers import load_weights
from src.training_helpers import set_seed

from src.trainer_and_evaluator import TrainerAndEvaluator
from src.tasks.train_and_evaluate import task_train_and_evaluate, task_config, setup_task, build_data_loaders

from collections import ChainMap
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch

from src.comet_logger import CometLogger
from src.settings import MODEL_CLASS
from src.model import find_model
from src.noise_evaluator import NoiseEvaluator

def build_test_data_loader(config, dataset_class, sampler_func):
    train_set, dev_set, test_set = dataset_class.splits(config)

    print("test set", len(test_set))

    sampler = sampler_func(train_set, config)
    return data.DataLoader(test_set,  num_workers=4,batch_size=1000)

def task_config(custom_config={}):
    """ Reasonable defaults for the training task """
    default_config = {
        'dev_every': 1,
        'model_class': MODEL_CLASS,
        'print_confusion_matrix': False,
    }
    return dict(ChainMap(custom_config, default_config))

def setup_task(config, n_labels):
    """ returns all the objects (including data loaders) for training """
    logger = CometLogger(project=config['project'])
    experiment = logger.experiment()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = find_model(config['model_class'], n_labels)
    return {
        'experiment': experiment,
        'model': model,
        'device': device,
        'config': config,
    }

def build_config():
    config = task_config({
            'project': 'chillanto-noise',
            'model_path': CHILLANTO_MODELS_FOLDER / 'chillanto_noise' ,
            'log_file_path': CHILLANTO_LOGGING_FOLDER ,
            'predictions_path': CHILLANTO_LOGGING_FOLDER ,
            "data_folder": CHILLANTO_DATA_FOLDER,
            'print_confusion_matrix': True,
            'input_length': 8000,
            'batch_size': 50,
            'wanted_words': ['normal', 'asphyxia'],
            "dev_pct": 5,
            "test_pct": 40,
            "sampling_freq": 8000,
            'model_class': 'res8',
            'timeshift_ms': 100,
            'use_nesterov': False,
            'n_epochs': None,
            'n_labels': 4,
            'schedule': [],
            'dev_every': None,
            'seed': 3,
            'cache_size':32768,
            })

    # Merge together the model, training and dataset configuration:
    return dict(ChainMap(ChillantoNoiseMixDataset.default_config(config), config))


def xxnoise_evaluate(label, tag, source_model_path):
    config = build_config()
    set_seed(config)
    data_loaders = build_data_loaders(config, ChillantoNoiseMixDataset, chillanto_sampler)
    params = setup_task(config, data_loaders, 4)
    experiment = params['experiment']
    te = TrainerAndEvaluator(params)
    te.model.load_state_dict(torch.load(source_model_path))
    te.set_best_model()
    te.evaluate()

def noise_evaluate(label, tag, source_model_path):

    config = build_config()
    config['label'] = label
    set_seed(config)
    config['bg_noise_files']= [ CHILLANTO_NOISE_DATA_FOLDER / 'gaussian_0_1_noise.wav']


    params = setup_task(config, 4)
    experiment = params['experiment']
    experiment.add_tag(tag)
    noise_range = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for pct in noise_range:
        config['noise_pct'] = pct
        experiment.log_metric('noise_pct', pct)
        test_data_loader = build_test_data_loader(config, ChillantoNoiseMixDataset, chillanto_sampler)
        te = NoiseEvaluator(params)
        te.test_loader = test_data_loader
        state_dict = torch.load(source_model_path)
        te.model.load_state_dict(state_dict)
        te.evaluate()
