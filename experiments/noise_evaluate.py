from pathlib import Path
from collections import ChainMap
from src.settings import CHILLANTO_DATA_FOLDER, CHILLANTO_LOGGING_FOLDER, CHILLANTO_MODELS_FOLDER, CHILLANTO_NOISE_DATA_FOLDER
from src.datasets.chillanto_noise_mix import ChillantoNoiseMixDataset
from src.training_helpers import set_seed
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

def build_test_data_loader(config, dataset_class):
    train_set, dev_set, test_set = dataset_class.splits(config)
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

def build_config(seed):
    config = task_config({
            'project': 'chillanto-noise-debug',
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
            'seed': seed,
            'cache_size':32768,
            })

    # Merge together the model, training and dataset configuration:
    return dict(ChainMap(ChillantoNoiseMixDataset.default_config(config), config))

def set_noise_files(noise_type):
    if noise_type == 'gaussian':
        return [ CHILLANTO_NOISE_DATA_FOLDER / 'gaussian_0_1_noise.wav' ]
    if noise_type == 'dog_bark':
        return [ CHILLANTO_NOISE_DATA_FOLDER / 'dog_bark' / f'{fn}.wav' for fn in [68]]
    if noise_type == 'children_playing':
        return [ CHILLANTO_NOISE_DATA_FOLDER / 'children_playing' / f'{fn}.wav' for fn in [54]]
    if noise_type == 'siren':
        return [ CHILLANTO_NOISE_DATA_FOLDER / 'siren' / f'{fn}.wav' for fn in [3]]

def load_model(model, source_model_path):
    state_dict = torch.load(source_model_path, map_location='cuda:0')

    desired_model_params = {}
    for (name, val) in state_dict.items():
        # remove the module. prefix that occurs with nn.data.Parallel
        name = name.replace('module.','')
        desired_model_params[name] = val
    model.load_state_dict(desired_model_params)
    return model

def noise_evaluate(noise_type, tag, source_model_path, seed=3, noise_range=[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):

    config = build_config(seed)
    config['noise_type'] = noise_type
    set_seed(config)

    config['noise_files']= set_noise_files(noise_type)
    print("noise!", noise_type, config['noise_files'])
    params = setup_task(config, 4)
    experiment = params['experiment']
    experiment.add_tag(tag)
    for pct in noise_range:
        config['noise_pct'] = float(pct)
        params['model'] = load_model(params['model'], source_model_path)
        experiment.log_metric('noise_pct', pct)
        test_data_loader = build_test_data_loader(config, ChillantoNoiseMixDataset)
        te = NoiseEvaluator(params)
        te.test_loader = test_data_loader
        te.evaluate()
