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
            'print_confusion_matrix': True,
            'n_epochs':100,
            'batch_size': 64,
            'cache_size':32768,
            'weight_decay': 0.00001,
            'dev_every': 1,
            'model_class': 'res8',
            })

    # Merge together the model, training and dataset configuration:
    return SpeechCommandsDataset.default_config(config)


def train_and_evaluate():
    config = build_config()
    print('Training on speech commands dataset...' +
        '\nModel={}\nno of epochs={}\nbatch size={}\ndev every={}'.format(
            config['model_class'], config['n_epochs'], config['batch_size'],
                config['dev_every']))
    data_loaders = build_data_loaders(config, SpeechCommandsDataset, speech_commands_sampler)
    params = setup_task(config, data_loaders, 12)
    task_train_and_evaluate(params)
