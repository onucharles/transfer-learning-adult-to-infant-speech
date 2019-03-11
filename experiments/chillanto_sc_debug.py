from pathlib import Path
from collections import ChainMap

from src.settings import CHILLANTO_DATA_FOLDER, CHILLANTO_LOGGING_FOLDER, CHILLANTO_MODELS_FOLDER
from src.datasets.chillanto import ChillantoDataset, chillanto_sampler
from src.tasks.train_and_evaluate import task_train_and_evaluate, task_config, setup_task, build_data_loaders
from src.training_helpers import load_weights
from src.training_helpers import set_seed
from src.trainer_and_evaluator import TrainerAndEvaluator

from src.datasets.speech_commands import SpeechCommandsDataset, speech_commands_sampler
from src.settings import SPEECH_COMMANDS_DATA_FOLDER, SPEECH_COMMANDS_LOGGING_FOLDER, SPEECH_COMMANDS_MODELS_FOLDER

params_to_load = ['conv0.weight', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked',
                       'conv1.weight', 'bn2.running_mean', 'bn2.running_var', 'bn2.num_batches_tracked',
                       'conv2.weight', 'bn3.running_mean', 'bn3.running_var', 'bn3.num_batches_tracked',
                       'conv3.weight', 'bn4.running_mean', 'bn4.running_var', 'bn4.num_batches_tracked',
                       'conv4.weight', 'bn5.running_mean', 'bn5.running_var', 'bn5.num_batches_tracked',
                       'conv5.weight', 'bn6.running_mean', 'bn6.running_var', 'bn6.num_batches_tracked',
                       'conv6.weight']

def build_sc_config():
    config = task_config({
            'project': 'sc_incremental',
            'model_path': SPEECH_COMMANDS_MODELS_FOLDER  ,
            'log_file_path': SPEECH_COMMANDS_LOGGING_FOLDER ,
            'predictions_path': SPEECH_COMMANDS_LOGGING_FOLDER ,
            'data_folder': SPEECH_COMMANDS_DATA_FOLDER,
            'print_confusion_matrix': False,
            'n_epochs':26,
            'batch_size': 64,
            "n_feature_maps": 45,
            "unknown_prob": 0.1,
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
            "test_pct": 10,
            "timeshift_ms": 100,
            "train_pct": 80,
            })

    # Merge together the model, training and dataset configuration:
    return SpeechCommandsDataset.default_config(config)



def sc_debug():


    sc_config = build_sc_config()
    set_seed(sc_config)
    sc_data_loaders = build_data_loaders(sc_config, SpeechCommandsDataset, speech_commands_sampler)
    sc_params = setup_task(sc_config, sc_data_loaders, 12)
    experiment = sc_params['experiment']
    good_checkpoint = '/network/home/maloneyj/saved_sc_model.pt'
    sc_params['model'] = load_weights(sc_params['model'], good_checkpoint, params_to_load)
    te = TrainerAndEvaluator(sc_params)
    te.perform_training()
