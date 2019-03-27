from pathlib import Path
from src.settings import CHILLANTO_DATA_FOLDER, CHILLANTO_LOGGING_FOLDER, CHILLANTO_MODELS_FOLDER
from src.datasets.chillanto import ChillantoDataset, chillanto_sampler
from src.tasks.train_and_evaluate import task_train_and_evaluate, task_config, setup_task, build_data_loaders
from src.training_helpers import set_seed
def build_config(seed):
    config = task_config({
            'project': 'chillanto2',
            'model_path': CHILLANTO_MODELS_FOLDER / 'chillanto' ,
            'log_file_path': CHILLANTO_LOGGING_FOLDER ,
            'predictions_path': CHILLANTO_LOGGING_FOLDER ,
            "data_folder": CHILLANTO_DATA_FOLDER,
            'print_confusion_matrix': True,
            'lr': [0.001, 0.0001],
            'weight_decay': 0.00001,
            'momentum': 0.9,
            'schedule': [5400],
            'n_epochs': 32,
            'n_labels': 4,
            'silence_prob': 0.0,
            'noise_prob': 0.0,
            'unknown_prob': 0.0,
            'input_length': 8000,
            'wanted_words': ['normal', 'asphyxia'],
            'batch_size': 32,
            "dev_pct": 5,
            "test_pct": 40,
            "sampling_freq": 8000,
            'model_class': 'res8',
            'timeshift_ms': 100,
            'use_nesterov': False,
            'seed': seed,
            })

    # Merge together the model, training and dataset configuration:
    return ChillantoDataset.default_config(config)


def train_and_evaluate(seed=3):
    config = build_config(seed)
    set_seed(config)
    data_loaders = build_data_loaders(config, ChillantoDataset, chillanto_sampler)
    params = setup_task(config, data_loaders, 4)
    task_train_and_evaluate(params)
