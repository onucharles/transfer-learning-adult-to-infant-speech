"""

Dataset: Speech Commands Dataset
Task: Save speech embeddings


"""

from src.settings import SPEECH_COMMANDS_DATA_FOLDER, SPEECH_COMMANDS_EMBEDDINGS_FOLDER
from src.comet_logger import CometLogger
from pathlib import Path
from src.model import find_model
from src.datasets.speech_commands import SpeechCommandsDataset
import numpy as np
import torch.utils.data as data
import torch
from torch import tensor

from sklearn.externals import joblib
from collections import ChainMap

def save_embeddings():
    logger = CometLogger(project='sc01_save_embeddings')

    config = SpeechCommandsDataset.default_config({"data_folder": SPEECH_COMMANDS_DATA_FOLDER})

    _, _, test_set = SpeechCommandsDataset.splits(config)
    test_loader = data.DataLoader(test_set, batch_size=min(len(test_set), 1))

    model = find_model(MODEL_CLASS, 12)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    embeddings_list = []
    labels_list = []
    experiment = logger.experiment()
    experiment.log_parameters(dict(ChainMap(config, { 'model_class': 'res8' })))
    for model_in, labels in test_loader:
        model_in = model_in.to(device)
        labels = labels.cpu().numpy()
        embedding = model.get_embedding(model_in).detach().cpu().numpy().flatten()
        embeddings_list.append(embedding)
        labels_list.append(int(labels))
        experiment.log_metric('embedding', int(labels))
#        print('embedding dim {0} and labels dim {1}'.format(embedding.shape, labels.shape))

    embeddings = np.array(embeddings_list)
    labels = np.array(labels_list)
    joblib.dump((embeddings, labels), SPEECH_COMMANDS_EMBEDDINGS_FOLDER / 'dump')
