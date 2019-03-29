from pathlib import Path
from collections import ChainMap
from src.settings import CHILLANTO_DATA_FOLDER, CHILLANTO_LOGGING_FOLDER, CHILLANTO_MODELS_FOLDER, CHILLANTO_NOISE_DATA_FOLDER
from src.datasets.chillanto_freq_mask import ChillantoFreqMaskDataset, chillanto_sampler
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
from src.freq_evaluator import FreqEvaluator

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

def build_config(seed):
    config = task_config({
            'project': 'chillanto-frequency-mask',
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
    return dict(ChainMap(ChillantoFreqMaskDataset.default_config(config), config))

def load_model(model, source_model_path):
    state_dict = torch.load(source_model_path)
    desired_model_params = {}
    for (name, val) in state_dict.items():
        # remove the module. prefix that occurs with nn.data.Parallel
        name = name.replace('module.','')
        desired_model_params[name] = val
    model.load_state_dict(desired_model_params)
    return model

def freq_ablation_evaluate(tag, source_model_path, seed=3):
    config = build_config(seed)
    set_seed(config)
    params = setup_task(config, 4)
    experiment = params['experiment']
    experiment.add_tag(tag)
    ranges = [ (0,10), (11,20), (21,30), (31, 40), (41, 50), (51, 60), (61,70), (71,80), (81,90), (91, 101)]
    for idx, freq_range in enumerate(ranges):
        params['model'] = load_model(params['model'], source_model_path)
        config['freq_range'] = freq_range
        experiment.log_metric('freq_range', str(freq_range))
        test_data_loader = build_test_data_loader(config, ChillantoFreqMaskDataset , chillanto_sampler)
        evaluator = FreqEvaluator(params)
        evaluator.test_loader = test_data_loader
        evaluator.step = idx + 1
        evaluator.evaluate()
