import os
import numpy as np
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
from pathlib import Path
from sklearn.externals import joblib
from src.training_helpers import print_eval, set_seed, compute_eval, confusion_matrix, print_f1_confusion_matrix, calc_f1_prec_recall


class NoiseEvaluator():
    def __init__(self, task_params):
        self.experiment = task_params['experiment']
        config = task_params['config']
        self.device = task_params['device']

        self.setup_model(task_params)

        self.test_loader = None

        self.n_epochs = config['n_epochs']
        self.schedule = config['schedule']
        self.n_labels = config['n_labels']
        self.dev_every = config['dev_every']
        self.print_confusion_matrix = config['print_confusion_matrix']

        self.setup_logs()
        self.setup_paths(config, self.experiment)
        self.experiment.log_parameters(config)
        self.config = config
        self.noise_pct = config['noise_pct'] * 100

        self.setup_loss_func(config)

    def setup_loss_func(self, config):
        self.loss_func = None

        if config["loss"] == "hinge":
            self.loss_func = nn.MultiMarginLoss()
        if config["loss"] == "crossent":
            self.loss_func = nn.CrossEntropyLoss()

    def setup_logs(self):
        self.train_logs = []
        self.valid_logs = []

    def setup_model(self, task_params):
        model = task_params['model']
        model.to(self.device)
        self.model = torch.nn.DataParallel(model)

    def setup_paths(self, config, experiment):
        model_path = config['model_path'] / experiment.id
        predictions_path = config['predictions_path'] / experiment.id
        log_file_path = config['log_file_path'] / experiment.id
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(predictions_path, exist_ok=True)
        os.makedirs(log_file_path, exist_ok=True)
        self.model_path = model_path
        self.predictions_path = f"{predictions_path}/predictions.pkl"
        self.log_file_path = f"{log_file_path}/training.log"

    def print_confusion_matrix_results(self, label, acc):
        if self.print_confusion_matrix:
            print_f1_confusion_matrix(label, acc, self.conf_mat)

    def report_f1_precision_recall(self, label):
        if self.print_confusion_matrix:
            f1, precision, recall = calc_f1_prec_recall(self.conf_mat)
            self.experiment.log_metric(f'{label}_F1', f1, step=self.noise_pct)
            self.experiment.log_metric(f'{label}_precision', precision,
                    step=self.noise_pct)
            self.experiment.log_metric(f'{label}_recall', recall,
                    step=self.noise_pct)

    def update_confusion_matrix(self, scores, labels):
        if self.print_confusion_matrix:
            self.conf_mat += confusion_matrix(scores.detach().cpu(), labels.detach().cpu(), np.arange(self.n_labels))

    def reset_confusion_matrix(self):
        if self.print_confusion_matrix:
            self.conf_mat = np.zeros((self.n_labels, self.n_labels))

    def dump_logs_and_predictions(self, predictions):
        joblib.dump((self.train_logs, self.valid_logs), self.log_file_path)
        print(f'Training logs were written to: {self.log_file_path}')
        print(f'Experiment: {self.experiment._get_experiment_url()}')
        joblib.dump(predictions, self.predictions_path)

    def evaluate(self):
        print("Evaluate")
        self.model.eval()
        criterion = self.loss_func
        results = []
        total = 0
        self.reset_confusion_matrix()
        prediction_log = []
        with torch.no_grad():
            for model_in, labels in self.test_loader:
                model_in = model_in.to(self.device)
                labels = labels.to(self.device)
                scores = self.model(model_in.clone().detach())
                loss = criterion(scores, labels)
                model_in_size = model_in.size(0)
                results.append(compute_eval(scores, labels) * model_in_size)
                self.update_confusion_matrix(scores, labels)
                total += model_in.cpu().size(0)
                prediction_log.append((scores, labels))
        acc = sum(results) / total

        print("Testing Accuracy", acc)
        self.experiment.log_metric('test_accuracy', acc)
        self.print_confusion_matrix_results("Testing", acc)
        self.report_f1_precision_recall('test')
        return prediction_log
