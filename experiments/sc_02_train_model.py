from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from lib.settings import MODEL_CLASS, SPEECH_COMMANDS_OUTPUT_FOLDER, SPEECH_COMMANDS_LOGGING_FOLDER, SPEECH_COMMANDS_MODELS_FOLDER
from lib.comet_logger import CometLogger
from pathlib import Path
from lib.model import find_model
from lib.speech_commands_dataset import SpeechCommandsDataset
import numpy as np
import torch.utils.data as data
import torch
from torch import tensor

from sklearn.externals import joblib
from collections import ChainMap, Counter
from lib.training_helpers import print_eval, set_seed, get_loss_fxn, get_sampler, evaluate, compute_eval, confusion_matrix, print_test_eval


def build_optimizer(model, config, lr):
    return torch.optim.SGD(model.parameters(), lr=lr, nesterov=config["use_nesterov"], weight_decay=config["weight_decay"], momentum=config["momentum"])

def train(config):

    logger = CometLogger(project='sc02_train')
    experiment = logger.experiment()
    ds_config = SpeechCommandsDataset.default_config({ })

    print(ds_config)
    train_set, dev_set, test_set = SpeechCommandsDataset.splits(ds_config)
    n_labels = ds_config["n_labels"]


    print("training set: ", len(train_set))
    print("dev set", len(dev_set))
    print("test set", len(test_set))


    #summary_writer = SummaryWriter(config['logs_dir'])
    # print('Logs will be written to: ', config['logs_dir'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = find_model(MODEL_CLASS)
    model.to(device)
    optimizers = [ build_optimizer(model, config, lr) for lr in config['lr'] ]
    optimizers.reverse()
    optimizer = optimizers.pop()
    print('opt', optimizer)
    criterion = get_loss_fxn(config)
    max_acc = 0

    sampler = get_sampler(train_set, ds_config)
    train_loader = data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=False, drop_last=True, sampler=sampler)
    dev_loader = data.DataLoader(dev_set, batch_size=min(len(dev_set), 16), shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=min(len(test_set), 16), shuffle=True)
    step_no = 0

    experiment.log_parameters(dict(ChainMap(ds_config, config, { 'model_class': MODEL_CLASS })))

    train_logs = []
    valid_logs = []
    for epoch_idx in range(config["n_epochs"]):
        print(f"Epoch {epoch_idx}")
        for batch_idx, (model_in, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            model_in = model_in.to(device)
            labels = labels.to(device)
            scores = model(model_in.clone().detach())
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            step_no += 1

            if step_no in config["schedule"]:
                optimizer = optimizers.pop()
                print("changing learning rate to {}".format(optimizer))
            train_score = compute_eval(scores, labels)
            print_eval("train step #{0} {1}".format(step_no, "accuracy"), train_score, loss, optimizer.defaults['lr'])
            train_logs.append((step_no, train_score.item(), loss.item()))

        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            model.eval()
            accs = []
            conf_mat = np.zeros((n_labels, n_labels))
            for model_in, labels in dev_loader:
                model_in = model_in.to(device)
                labels = labels.to(device)
                scores = model(model_in.clone().detach())
                loss = criterion(scores, labels)
                loss_numeric = loss.item()
                accs.append(compute_eval(scores, labels).detach().cpu())
                conf_mat += confusion_matrix(scores.detach().cpu(), labels.detach().cpu(), np.arange(n_labels))
            avg_acc = np.mean(accs)
            print_test_eval("Validation", avg_acc, conf_mat, config['compute_f1'])
            valid_logs.append((step_no, avg_acc, loss.item()))

            # save best model
            if avg_acc > max_acc:
                print("saving best model...")
                max_acc = avg_acc
                torch.save(model.state_dict(), config['model_file'])

    log_file_path = SPEECH_COMMANDS_LOGGING_FOLDER /  'logs.pkl'
    joblib.dump((train_logs, valid_logs), log_file_path)
    print('Training logs were written to: ', log_file_path)

    # Test model
    predictions = evaluate(config, model, device, test_loader)
    joblib.dump(predictions, SPEECH_COMMANDS_LOGGING_FOLDER / 'predictions.pkl')

def train_model():
    config = {
        'n_epochs': 1,
        'lr': [0.1, 0.01, 0.001],
        'schedule': [0, 3000, 6000],
        'batch_size': 64,
        'weight_decay': 0.00001,
        'model_file': SPEECH_COMMANDS_MODELS_FOLDER / 'latest.mdl',
        'dev_every': 1,
        'use_nesterov': False,
        'weight_decay': False,
        'momentum': False,
        'compute_f1': True,
    }

    train(config)
