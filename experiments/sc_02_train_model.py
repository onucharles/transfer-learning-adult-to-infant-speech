from pathlib import Path
import torch
import torch.utils.data as data
import numpy as np

from src.settings import MODEL_CLASS, SPEECH_COMMANDS_OUTPUT_FOLDER, SPEECH_COMMANDS_LOGGING_FOLDER, SPEECH_COMMANDS_MODELS_FOLDER
from src.comet_logger import CometLogger
from pathlib import Path
from src.model import find_model
from src.speech_commands_dataset import SpeechCommandsDataset

from sklearn.externals import joblib
from collections import ChainMap
from src.tasks.train_and_evaluate import task_train_and_evaluate


def get_sampler(train_set, config):
    # TODO fix this turkey
    # TODO need to also add support in 'config' for sampler, such that no sampler is also valid.
    # TODO  ideally get class prob from train_set
    # class_prob = [0, 0, 0.76, 0.24]
    #sample_weights = []

    sample_weights = np.zeros(len(train_set)) + (1 / config['n_labels'])
    #for i in range(len(train_set)):
    #    _, label = train_set[i]
    #    sample_weights.append(1 / class_prob[label])

    sample_weights = torch.tensor(sample_weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(train_set))

    return sampler

def build_data_loaders(config):
    train_set, dev_set, test_set = SpeechCommandsDataset.splits(config)

    print("training set: ", len(train_set))
    print("dev set", len(dev_set))
    print("test set", len(test_set))

    sampler = get_sampler(train_set, config)
    train= data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=False, drop_last=True, sampler=sampler)
    dev= data.DataLoader(dev_set, batch_size=min(len(dev_set), 16), shuffle=True)
    test= data.DataLoader(test_set, batch_size=min(len(test_set), 16), shuffle=True)
    return train, dev, test

def setup_and_run_training(config):
    logger = CometLogger(project='speech_commands_train')
    experiment = logger.experiment()
    train_loader, dev_loader, test_loader = build_data_loaders(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = find_model(MODEL_CLASS)

    return {
        'experiment': experiment,
        'model': model,
        'device': device,
        'config': config,
        'loaders': (train_loader, dev_loader, test_loader)
    }

def train_model():
    train_config = {
        'n_epochs': 1,
        'lr': [0.1, 0.01, 0.001],
        'schedule': [0, 3000, 6000],
        'batch_size': 256, # 64
        'weight_decay': 0.00001,
        'model_path': SPEECH_COMMANDS_MODELS_FOLDER / 'latest.mdl',
        'log_file_path': SPEECH_COMMANDS_LOGGING_FOLDER /  'logs.pkl',
        'predictions_path': SPEECH_COMMANDS_LOGGING_FOLDER / 'predictions.pkl',
        'dev_every': 1,
        'use_nesterov': False,
        'weight_decay': False,
        'momentum': False,
        'seed': 1,
    }
    ds_config = SpeechCommandsDataset.default_config({ })

    # Merge together the model, training and dataset configuration:
    config = dict(ChainMap(ds_config, train_config, { 'model_class': MODEL_CLASS }))

    params = setup_and_run_training(config)
    task_train_and_evaluate(params)
