from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data

from src.settings import SITW_DATA_FOLDER, SITW_MODELS_FOLDER, SITW_LOGGING_FOLDER
from src.datasets.sitw import SITWDataset
from src.tasks.train_and_evaluate import task_train_and_evaluate, task_config, setup_task
from src.training_helpers import set_seed

def build_data_loaders(config, splits):
    train_set, dev_set, test_set = splits['datasets']
    print("training set: ", len(train_set))
    print("dev set", len(dev_set))
    print("test set", len(test_set))
    train= data.DataLoader(train_set, num_workers=8, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    dev= data.DataLoader(dev_set,  num_workers=8, batch_size=min(len(dev_set), 64), shuffle=True)
    test= data.DataLoader(test_set,num_workers=8,  batch_size=min(len(test_set), 64), shuffle=True)
    return train, dev, test

def build_config():
    config = task_config({
            'project': 'sitw_train_and_evaluate',
            'model_path': SITW_MODELS_FOLDER ,
            'log_file_path': SITW_LOGGING_FOLDER ,
            'predictions_path': SITW_LOGGING_FOLDER ,
            'data_folder': SITW_DATA_FOLDER,
            'print_confusion_matrix': False,
            'n_epochs': 32,
            'lr': [0.01, 0.001, 0.0001],
            'schedule': [1500, 3000],
            'batch_size': 50,
            'weight_decay': 0.00001,
            'momentum': 0.9,
            'label_limit': False,
            'seed': 9,
            'model_class': 'res8',
            'input_length': 16000,
            'loss': 'hinge',
            })
    # Merge together the model, training and dataset configuration:
    return SITWDataset.default_config(config)


def train_and_evaluate():
    config = build_config()
    set_seed(config)
    splits = SITWDataset.splits(config)
    config['n_labels'] = splits['n_labels']
    data_loaders = build_data_loaders(config, splits)
    params = setup_task(config, data_loaders, config['n_labels'])



    # print number of parameters in model
    model, experiment = params['model'], params['experiment']
    no_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model has {0} parameters'.format(no_of_params))
    experiment.set_model_graph((str(model)))
    task_train_and_evaluate(params)
