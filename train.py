"""
MIT License

Copyright (c) 2018 Castorini

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Speech command sample script:
python train.py \
--data_folder /mnt/hdd/Datasets/speech-commands-8k-16bit \
--output_folder /mnt/hdd/Experiments/speech-commands-pytorch \
--input_length 8000 \
--wanted_words yes no up down left right on off stop go \
--dev_every 1 \
--n_epochs 1 \
--batch_size 64 \
--weight_decay 0.00001 \
--lr 0.1 0.01 0.001 \
--schedule 3000 6000 \
--model res8 \
--sampling_freq 8000 \
--dev_pct 10 \
--test_pct 10 \
--silence_prob 0.1 \
--unknown_prob 0.1 \
--noise_prob 0.8 \
--timeshift_ms 100 \
--gpu_no 1 \
--input_file /mnt/hdd/Experiments/speech-commands-pytorch/20181022-160051/model.pt \
--mode eval

Chillanto sample script:
python train.py \
--data_folder /mnt/hdd/Datasets/chillanto-8k-16bit-renamed \
--output_folder /mnt/hdd/Experiments/chillanto-pt \
--input_length 8000 \
--wanted_words normal asphyxia \
--n_labels 4 \
--dev_every 1 \
--n_epochs 50 \
--batch_size 50 \
--weight_decay 0.00001 \
--lr 0.001 0.0001 \
--schedule 500  \
--model res8 \
--sampling_freq 8000 \
--dev_pct 5 \
--test_pct 40 \
--silence_prob 0.0 \
--unknown_prob 0.0 \
--noise_prob 0.0 \
--timeshift_ms 100 \
--compute_f1 true \
--gpu_no 1 \
--input_file /mnt/hdd/Experiments/chillanto-pt/20181025-114623/model.pt \
--no_cuda true

--input_file /mnt/hdd/Experiments/speech-commands-pytorch/20181022-160051/model.pt \
"""

from collections import ChainMap
import argparse
import os
import random
import sys

from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

import model as mod
from utils.ioutils import load_json, save_json, create_folder, current_datetime
from utils import evalutils
from sklearn import metrics
from tensorboardX import SummaryWriter
from sklearn.externals import joblib
from ConfigBuilder import ConfigBuilder

def prepare_experiment_directory(config):
    """ Sets up folders where all experiment files will be saved to.
    """
    # create unique sub-folder for this experiment
    exp_dir = config['output_folder'] + '/' + current_datetime() + '/'
    create_folder(exp_dir)
    print('Experiment files will be saved to: ', exp_dir)

    # Specify full path to model file.
    config["model_file"] = exp_dir + "model.pt"

    # Specify logs directory
    config['exp_dir'] = exp_dir
    config['logs_dir'] = exp_dir + 'logs/'
    create_folder(config['logs_dir'])       # TODO remove this if using tensorboard to write logs.

    # save parameters to json file
    config_path = exp_dir + 'config.json'
    save_json(dict(config), config_path)
    print('Saved experiment parameters to: ', config_path)

def compute_eval(scores, labels):
    batch_size = labels.size(0)
    predictions = torch.max(scores, 1)[1].view(batch_size).data

    metric_value = (predictions == labels.data).float().sum() / batch_size
    return metric_value

def confusion_matrix(scores, labels, classes=None):
    batch_size = labels.size(0)
    predictions = torch.max(scores, 1)[1].view(batch_size).data
    cm = metrics.confusion_matrix(labels.data, predictions, labels=classes)
    return cm

def print_eval(name, eval_score, loss, lr,  end="\n"):
    print("{0}: {1}, loss: {2}, learning_rate: {3}".format(name, eval_score, loss.item(), lr), end=end)

def print_test_eval(name, avg_acc, conf_mat, compute_f1=False):
    if compute_f1:
        tp, tn, fp, fn, p, n = evalutils.read_conf_matrix(conf_mat, pos_class=3)
        f1, precision, recall = evalutils.f1_prec_recall(tp, tn, fp, fn, p, n)
        print("Confusion matrix: {}".format(conf_mat))
        print("{} accuracy: {}\tF1 = {}\tPrecision={}\tRecall={}".format(name, avg_acc, f1, precision, recall))
    else:
        print("{} accuracy: {}".format(name, avg_acc))

def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def evaluate(config, model=None, test_loader=None):
    n_labels = config["n_labels"]
    classes = np.arange(n_labels)
    if not test_loader:
        _, _, test_set = mod.SpeechDataset.splits(config)
        test_loader = data.DataLoader(test_set, batch_size=min(len(test_set), 1))
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
    if not model:
        model = config["model_class"](config)
        # model.load(config["input_file"])
        load_weights(model, config['input_file'], config['params_to_load'])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()
    criterion = nn.CrossEntropyLoss()
    results = []
    total = 0
    conf_mat = np.zeros((n_labels,n_labels))
    prediction_log = []
    for model_in, labels in test_loader:
        model_in = Variable(model_in, requires_grad=False)
        if not config["no_cuda"]:
            model_in = model_in.cuda()
            labels = labels.cuda()
        scores = model(model_in)
        labels = Variable(labels, requires_grad=False)
        loss = criterion(scores, labels)
        results.append(compute_eval(scores.detach().cpu(), labels.detach().cpu()) * model_in.size(0))
        total += model_in.size(0)
        conf_mat += confusion_matrix(scores.detach().cpu(), labels.detach().cpu(), classes)
        prediction_log.append((scores, labels))
    acc = sum(results) / total
    print_test_eval("Testing", acc, conf_mat, config['compute_f1'])
    joblib.dump(prediction_log, config['exp_dir'] + 'predictions.pkl')

def load_weights(model, state_dict_path, params_to_load=[]):
    """ Load weights into model

    Args
        model - an instance of the model
        state_dict_path - string path to state_dict to be loaded
        params_to_load - list of parameter names to be loaded.
    """
    print("loading weights from model at '{0}'.\nParameters to load are: {1}".format(state_dict_path, params_to_load))
    state_dict = torch.load(state_dict_path)

    desired_state_dict = {}
    for name, item in state_dict.items():
        if name in params_to_load:
            desired_state_dict[name] = item

    model.load_state_dict(desired_state_dict, strict=False)

def get_loss_fxn(config):
    #criterion = nn.CrossEntropyLoss(weight=1/torch.cuda.FloatTensor([0, 0, 0.76, 0.24]))
    if config['loss'] == 'crossent':
        criterion = nn.CrossEntropyLoss()
    elif config['loss'] == 'hinge':
        criterion = nn.MultiMarginLoss()
    else:
        raise ValueError("Unsupported loss function '{0}' was specified".format(config['loss']))
    return criterion

def get_sampler(train_set, config):
    # TODO need to also add support in 'config' for sampler, such that no sampler is also valid.

    # TODO  ideally get class prob from train_set
    class_prob = [0, 0, 0.76, 0.24]
    sample_weights = []

    for i in range(len(train_set)):
        _, label = train_set[i]
        sample_weights.append(1 / class_prob[label])

    sample_weights = torch.Tensor(sample_weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(train_set))

    return sampler

def train(config):
    train_set, dev_set, test_set = mod.SpeechDataset.splits(config)
    n_labels = config["n_labels"]
    classes = np.arange(n_labels)
    print("training set: ", len(train_set))
    print("dev set", len(dev_set))
    print("test set", len(test_set))

    # summary_writer = SummaryWriter(config['logs_dir'])
    # print('Logs will be written to: ', config['logs_dir'])

    model = config["model_class"](config)
    if config["input_file"]:
        # model.load(config["input_file"])
        load_weights(model, config['input_file'], config['params_to_load'])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][0], nesterov=config["use_nesterov"], weight_decay=config["weight_decay"], momentum=config["momentum"])
    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    criterion = get_loss_fxn(config)
    max_acc = 0

    sampler = None #get_sampler(train_set, config)
    do_shuffle = True if sampler is None else False
    train_loader = data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=do_shuffle,
            drop_last=True, sampler=sampler)
    dev_loader = data.DataLoader(dev_set, batch_size=min(len(dev_set), 16), shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=min(len(test_set), 16), shuffle=True)
    step_no = 0

    train_logs = []
    valid_logs = []
    for epoch_idx in range(config["n_epochs"]):
        for batch_idx, (model_in, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            if not config["no_cuda"]:
                model_in = model_in.cuda()
                labels = labels.cuda()
            model_in = Variable(model_in, requires_grad=False)
            scores = model(model_in)
            labels = Variable(labels, requires_grad=False)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            step_no += 1
            if step_no > schedule_steps[sched_idx]: # TODO need to update optimiser here.
                sched_idx += 1
                print("changing learning rate to {}".format(config["lr"][sched_idx]))
                optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][sched_idx],
                    nesterov=config["use_nesterov"], momentum=config["momentum"], weight_decay=config["weight_decay"])
            train_score = compute_eval(scores, labels)
            print_eval("train step #{0} {1}".format(step_no, "accuracy"), train_score, loss,
                    optimizer.defaults['lr'])
            # summary_writer.add_scalar('data/train_acc', train_score.item(), epoch_idx)
            train_logs.append((step_no, train_score.item(), loss.item()))

        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            model.eval()
            accs = []
            conf_mat = np.zeros((n_labels, n_labels))
            for model_in, labels in dev_loader:
                batch_size = labels.size(0)
                set_size = len(dev_loader)
                model_in = Variable(model_in, requires_grad=False)
                if not config["no_cuda"]:
                    model_in = model_in.cuda()
                    labels = labels.cuda()
                scores = model(model_in)
                labels = Variable(labels, requires_grad=False)
                loss = criterion(scores, labels)
                loss_numeric = loss.item()
                accs.append(compute_eval(scores.detach().cpu(), labels.detach().cpu()))
                conf_mat += confusion_matrix(scores.detach().cpu(), labels.detach().cpu(), np.arange(n_labels))
            avg_acc = np.mean(accs)
            print_test_eval("Validation", avg_acc, conf_mat, config['compute_f1'])
            # summary_writer.add_scalar('data/valid_acc', avg_acc, epoch_idx)
            valid_logs.append((step_no, avg_acc, loss.item()))

            # save best model
            if avg_acc > max_acc:
                print("saving best model...")
                max_acc = avg_acc
                # model.save(config["model_file"])
                torch.save(model.state_dict(), config['model_file'])
    # verify parameters
    # state_dict = torch.load(config['input_file'])
    # model_state_dict = model.state_dict()
    # for param in config['params_to_load']:
    #     print('checking param ', param)
    #     print(torch.equal(state_dict[param], model_state_dict[param]))
    #
    # for param in config['params_to_update']:
    #     print('checking param ', param)
    #     print(torch.equal(state_dict[param], model_state_dict[param]))

    # summary_writer.close()
    log_file_path = config['logs_dir'] + 'logs.pkl'
    joblib.dump((train_logs, valid_logs), log_file_path)
    print('Training logs were written to: ', log_file_path)

    # Test model
    evaluate(config, model, test_loader)

def save_embeddings(config, model=None, test_loader=None):
    n_labels = config["n_labels"]
    classes = np.arange(n_labels)
    if not test_loader:
        _, _, test_set = mod.SpeechDataset.splits(config)
        test_loader = data.DataLoader(test_set, batch_size=min(len(test_set), 1))
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
    if not model:
        model = config["model_class"](config)
        load_weights(model, config['input_file'], config['params_to_load'])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()

    embeddings_list = []
    labels_list = []
    for model_in, labels in test_loader:
        model_in = Variable(model_in, requires_grad=False)
        if not config["no_cuda"]:
            model_in = model_in.cuda()
            labels = labels.cuda()
        labels = Variable(labels, requires_grad=False).numpy()
        embedding = model.get_embedding(model_in).detach().numpy().flatten()
        embeddings_list.append(embedding)
        labels_list.append(int(labels))

        # print('embedding dim {0} and labels dim {1}'.format(embedding.shape, labels.shape))

    embeddings = np.array(embeddings_list)
    labels = np.array(labels_list)
    print('embedding shape is {0} and labels shape is {1}'.format(embeddings.shape, labels.shape))
    joblib.dump((embeddings, labels), '/mnt/hdd/Dropbox (NRP)/paper/tsne_data/output_embeddings_transfer.pkl')


def main():
    # output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model", "model.pt")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[x.value for x in list(mod.ConfigType)], default="cnn-trad-pool2", type=str)
    config, _ = parser.parse_known_args()

    # Default to weights for 'res8' model
    params_to_load = ['conv0.weight', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked',
                       'conv1.weight', 'bn2.running_mean', 'bn2.running_var', 'bn2.num_batches_tracked',
                       'conv2.weight', 'bn3.running_mean', 'bn3.running_var', 'bn3.num_batches_tracked',
                       'conv3.weight', 'bn4.running_mean', 'bn4.running_var', 'bn4.num_batches_tracked',
                       'conv4.weight', 'bn5.running_mean', 'bn5.running_var', 'bn5.num_batches_tracked',
                       'conv5.weight', 'bn6.running_mean', 'bn6.running_var', 'bn6.num_batches_tracked',
                       'conv6.weight']
    #params_to_update = ['output.weight', 'output.bias']
    global_config = dict(no_cuda=False, n_epochs=26, lr=[0.001], schedule=[np.inf], batch_size=64, dev_every=10,
        seed=11, use_nesterov=False, input_file="", gpu_no=1, cache_size=32768, momentum=0.9, weight_decay=0.00001,
        output_folder="", compute_f1=False, params_to_load=params_to_load, loss='crossent')


    config = load_json('./sc_legacy.json')


    # Fetch model parameters, speech dataset parameters and global parameters.
    # Combine them with current config into ChainMap.
    builder = ConfigBuilder(
#        vars(config),
        config,
        mod.find_config(config['model']),
        mod.SpeechDataset.default_config(),
        global_config)
    parser = builder.build_argparse()
    #parser.add_argument("--mode", choices=["train", "eval", "save_embedding"], default="train", type=str)
    config = builder.config_from_argparse(parser)

    # Prepare experiment directory, save config to file.
    prepare_experiment_directory(config)

    # Add 'model_class' after, since it cannot be serialised and we don't need to anyways.
    config["model_class"] = mod.find_model(config["model"])

    # Run training or evaluation.
    set_seed(config)
    train(config)
    return
    if config["mode"] == "train":
        print("Running in training mode...")
        train(config)
    elif config["mode"] == "eval":
        print("Running in evaluation mode...")
        evaluate(config)
    elif config["mode"] == "save_embedding":
        print("Running in save embedding mode...")
        save_embeddings(config)

if __name__ == "__main__":
    main()
