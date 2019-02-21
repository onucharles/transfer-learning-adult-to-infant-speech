from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data

from src.settings import VOX_DATA_FOLDER, VOX_MODELS_FOLDER, VOX_LOGGING_FOLDER
from src.datasets.voxceleb_one import VoxCelebOneDataset
from src.tasks.train_and_evaluate import task_train_and_evaluate, task_config, setup_task
from src.training_helpers import set_seed

def voxceleb_sampler(distribution, total, labels, train_set):
    weights = np.zeros(len(labels))
    for (lbl, idx) in labels.items():
        if lbl in distribution:
            weights[idx] = distribution[lbl] / total
    return torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_set))

def build_data_loaders(config, splits):
    train_set, dev_set, test_set = splits['datasets']
    print("training set: ", len(train_set))
    print("dev set", len(dev_set))
    print("test set", len(test_set))
    sampler = voxceleb_sampler(splits['distribution'], splits['total'], splits['labels'], train_set)
    train= data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=False,
            num_workers=6,
            sampler=sampler)
#    train= data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    dev= data.DataLoader(dev_set,  num_workers=6, batch_size=min(len(dev_set),
        16), shuffle=True)
    test= data.DataLoader(test_set,num_workers=6,  batch_size=min(len(test_set),
        16), shuffle=True)
    return train, dev, test

def build_config():
    config = task_config({
            'project': 'voxceleb1_train_and_evaluate',
            'model_path': VOX_MODELS_FOLDER / 'latest.mdl',
            'log_file_path': VOX_LOGGING_FOLDER /  'logs.pkl',
            'predictions_path': VOX_LOGGING_FOLDER / 'predictions.pkl',
            'data_folder': VOX_DATA_FOLDER,
            'print_confusion_matrix': False,
            'n_epochs': 20,
            'lr': [0.00001],
            'schedule': [],
            'batch_size': 8,
            'model_class': 'res15',
            'weight_decay': 0.000001,
            'momentum': 0.9,
            'input_length': 40000,
            'label_limit': 10,
            'seed': 9
            })
    # Merge together the model, training and dataset configuration:
    return VoxCelebOneDataset.default_config(config)


def train_and_evaluate():
    config = build_config()
    set_seed(config)
    splits = VoxCelebOneDataset.splits(config)
    config['n_labels'] = splits['n_labels']
    data_loaders = build_data_loaders(config, splits)
    params = setup_task(config, data_loaders, config['n_labels'])
    task_train_and_evaluate(params)
