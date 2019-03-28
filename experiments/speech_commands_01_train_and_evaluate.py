from pathlib import Path
from src.settings import SPEECH_COMMANDS_DATA_FOLDER, SPEECH_COMMANDS_LOGGING_FOLDER, SPEECH_COMMANDS_MODELS_FOLDER
from src.datasets.speech_commands import SpeechCommandsDataset, speech_commands_sampler
from src.tasks.train_and_evaluate import task_train_and_evaluate, task_config, setup_task, build_data_loaders

def build_config():
    config = task_config({
            'project': 'speech_commands_train_and_evaluate',
            'model_path': SPEECH_COMMANDS_MODELS_FOLDER  ,
            'log_file_path': SPEECH_COMMANDS_LOGGING_FOLDER ,
            'predictions_path': SPEECH_COMMANDS_LOGGING_FOLDER ,
            'data_folder': SPEECH_COMMANDS_DATA_FOLDER,
            'print_confusion_matrix': False,
            'n_epochs':26,
            'batch_size': 64,
            "use_dilation": False,
            "use_nesterov": False,
            "lr": [0.1, 0.01, 0.001],
            "schedule": [ 3000.0, 6000.0 ],
            'cache_size':32768,
            "sampling_freq": 8000,
            "group_speakers_by_id": True,
            'weight_decay': 0.00001,
            'dev_every': 1,
            'model_class': 'res8',
            "dev_pct": 10,
            "silence_prob": 0.1,
            "unknown_prob": 0.1,
            "noise_prob": 0.8,
            "momentum": 0.9,
            "n_dct_filters": 40,
            "n_feature_maps": 45,
            "n_labels": 12,
            "n_layers": 6,
            "n_mels": 40,
            "seed": 10,
            "test_pct": 10,
            "timeshift_ms": 100,
            "train_pct": 80,
            "use_dilation": False,
            "use_nesterov": False,
            "loss": "crossent"
            })

    # Merge together the model, training and dataset configuration:
    return SpeechCommandsDataset.default_config(config)


def train_and_evaluate():
    config = build_config()
    print('Training on speech commands dataset...' +
        '\nModel={}\nno of epochs={}\nbatch size={}\ndev every={}'.format(
            config['model_class'], config['n_epochs'], config['batch_size'],
                config['dev_every']))
    data_loaders = build_data_loaders(config, SpeechCommandsDataset)
    params = setup_task(config, data_loaders, 12)
    task_train_and_evaluate(params)
