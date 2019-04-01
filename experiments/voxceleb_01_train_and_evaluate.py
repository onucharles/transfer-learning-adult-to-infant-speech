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
#    train= data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=False,
#            num_workers=6,
#            sampler=sampler)
    train= data.DataLoader(train_set, num_workers=8, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    dev= data.DataLoader(dev_set,  num_workers=8, batch_size=min(len(dev_set),
        64), shuffle=False)
    test= data.DataLoader(test_set,num_workers=8,  batch_size=min(len(test_set),
        64), shuffle=False)
    return train, dev, test

def build_config():
    config = task_config({
            'project': 'voxceleb1_train_and_evaluate',
            'model_path': VOX_MODELS_FOLDER ,
            'log_file_path': VOX_LOGGING_FOLDER ,
            'predictions_path': VOX_LOGGING_FOLDER ,
            'data_folder': VOX_DATA_FOLDER,
            'print_confusion_matrix': False,
            'n_epochs': 26,
            'lr': [0.001, 0.0001],
            'schedule': [250, 5000],
            'batch_size': 64,
            'weight_decay': 0.00001,
            'momentum': 0.9,
            'label_limit': 25,
            'seed': 9,
            'model_class': 'res8',
            'input_length': 8000,
            'loss': 'hinge',
            'timeshift_ms': 1000
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

    # print number of parameters in model
    model, experiment = params['model'], params['experiment']
    no_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model has {0} parameters'.format(no_of_params))
    experiment.set_model_graph((str(model)))
    task_train_and_evaluate(params)
