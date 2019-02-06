from pathlib import Path


MODEL_CLASS='res8'


DATA_FOLDER = Path('/').parent / 'network' / 'data1' / 'maloneyj'
SPEECH_COMMANDS_DATA_FOLDER = DATA_FOLDER / 'dataset_speech_commands'
SPEECH_COMMANDS_OUTPUT_FOLDER = DATA_FOLDER / 'dataset_speech_output'
SPEECH_COMMANDS_LOGGING_FOLDER = DATA_FOLDER / 'logs' / 'speech_commands'
SPEECH_COMMANDS_MODELS_FOLDER = DATA_FOLDER / 'models' / 'speech_commands'

COMET_API_KEY = 'bSyRm6vJpAwfehizXic7Fo0bY'
COMET_WORKSPACE = 'jlebensold'
