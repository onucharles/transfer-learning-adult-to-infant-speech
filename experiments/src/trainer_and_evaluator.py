import os
import numpy as np
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
from pathlib import Path
from sklearn.externals import joblib
from src.training_helpers import print_eval, set_seed, compute_eval, confusion_matrix, print_f1_confusion_matrix, calc_f1_prec_recall


class TrainerAndEvaluator():
    def __init__(self, task_params):
        self.task_params = task_params
        self.experiment = task_params['experiment']
        config = task_params['config']
        self.train_loader, self.dev_loader, self.test_loader = task_params['loaders']
        self.device = task_params['device']

        self.setup_model(task_params)

        self.epoch_no = None
        self.step_no = None
        self.max_eval_metric = None

        self.n_epochs = config['n_epochs']
        self.schedule = config['schedule']
        self.n_labels = config['n_labels']
        self.dev_every = config['dev_every']
        self.print_confusion_matrix = config['print_confusion_matrix']

        self.eval_metric = 'acc'
        if self.print_confusion_matrix:
            self.eval_metric = 'f1'

        self.setup_logs()
        self.setup_paths(config, self.experiment)
        self.experiment.log_parameters(config)
        self.config = config

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
        # trying for Speech commands to see if it makes a difference on
        # transfer:
        self.model = model
        #self.model = torch.nn.DataParallel(model)

    def load_best_model(self):
        state_dict_path = f"{self.model_path}/_best.mdl"
        print("loading ", state_dict_path)
        state_dict = torch.load(state_dict_path, map_location='cuda:0')
        self.model.load_state_dict(state_dict)

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

    def report_training(self, train_score, loss):
        print("Training - epoch:{} step:{} accuracy:{} loss:{} learning_rate:{}"
            .format(self.epoch_no, self.step_no, train_score, loss, self.optimizer.defaults['lr']))
        self.train_logs.append((self.step_no, train_score.item(), loss.item()))
        self.experiment.log_metric('train_loss', loss.item(), self.step_no)

    def print_confusion_matrix_results(self, label, acc):
        if self.print_confusion_matrix:
            print_f1_confusion_matrix(label, acc, self.conf_mat)

    def report_f1_precision_recall(self, label):
        f1 = None
        if self.print_confusion_matrix:
            f1, precision, recall = calc_f1_prec_recall(self.conf_mat)
            self.experiment.log_metric(f'{label}_F1', f1)
            self.experiment.log_metric(f'{label}_precision', precision)
            self.experiment.log_metric(f'{label}_recall', recall)
        return f1

    def update_confusion_matrix(self, scores, labels):
        if self.print_confusion_matrix:
            self.conf_mat += confusion_matrix(scores.cpu(), labels.cpu(), np.arange(self.n_labels))

    def reset_confusion_matrix(self):
        if self.print_confusion_matrix:
            self.conf_mat = np.zeros((self.n_labels, self.n_labels))

    def update_eval_metric_and_save_model(self, metric):
        # save best model
        if metric > self.max_eval_metric:
            print("saving best model...")
            self.max_eval_metric = metric
            torch.save(self.model.state_dict(),f"{self.model_path}/_best.mdl")

    def dump_logs_and_predictions(self, predictions):
        joblib.dump((self.train_logs, self.valid_logs), self.log_file_path)
        print(f'Training logs were written to: {self.log_file_path}')
        print(f'Experiment: {self.experiment._get_experiment_url()}')
        joblib.dump(predictions, self.predictions_path)

    def evaluate(self):
        print("Evaluate")
        criterion = self.loss_func
        results = []
        total = 0
        self.reset_confusion_matrix()
        prediction_log = []
        self.load_best_model()
        self.model.eval()
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

    def perform_training(self):
        self.epoch_no = 0
        self.step_no = 0
        self.max_eval_metric = 0.

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config["lr"][0],
                nesterov=self.config["use_nesterov"],
                weight_decay=self.config["weight_decay"],
                momentum=self.config["momentum"])
        criterion = self.loss_func
        sched_idx = 0
        schedule_steps = self.config["schedule"]
        schedule_steps.append(np.inf)
        for epoch_idx in range(self.n_epochs):
            print(f"Epoch {epoch_idx}")
            self.epoch_no += 1
            self.experiment.log_current_epoch(self.epoch_no)

            training_accuracies = []
            for batch_idx, (model_in, labels) in enumerate(self.train_loader):
                self.model.train()
                self.optimizer.zero_grad()
                self.step_no += 1

                model_in = model_in.to(self.device)
                labels = labels.to(self.device)

                model_in = Variable(model_in, requires_grad=False)
                scores = self.model(model_in)
                labels = Variable(labels, requires_grad=False)
                loss = criterion(scores, labels)
                loss.backward()

                self.optimizer.step()
                if self.step_no > schedule_steps[sched_idx]: # TODO need to update optimiser here.
                    sched_idx += 1
                    print("changing learning rate to {}".format(self.config["lr"][sched_idx]))
                    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config["lr"][sched_idx],
                        nesterov=self.config["use_nesterov"],
                        momentum=self.config["momentum"], weight_decay=self.config["weight_decay"])
                    self.experiment.log_metric('learning_rate', self.config["lr"][sched_idx], step=self.step_no)

                train_score = compute_eval(scores, labels)
                training_accuracies.append(train_score.cpu())
                self.report_training(train_score, loss)
            avg_train_acc = np.mean(training_accuracies)
            self.experiment.log_metric('train_accuracy', avg_train_acc, step=self.step_no)

            if epoch_idx % self.dev_every == self.dev_every - 1:
                print("Validation")
                self.model.eval()
                val_acc = 0
                val_loss = 0
                val_total_size = 0
                self.reset_confusion_matrix()
                with torch.no_grad():
                    for model_in, labels in self.dev_loader:
                        model_in = model_in.to(self.device)
                        labels = labels.to(self.device)
                        model_in = Variable(model_in, requires_grad=False)
                        scores = self.model(model_in)
                        labels = Variable(labels, requires_grad=False)
                        model_in_size = model_in.size(0)
                        val_loss += criterion(scores, labels) * model_in_size
                        valid_scores = compute_eval(scores.cpu(), labels.cpu())
                        # validation_accuracies.append(valid_scores.detach().cpu())
                        val_acc += valid_scores * model_in_size
                        val_total_size += model_in_size
                        self.update_confusion_matrix(scores, labels)

                # avg_dev_acc = np.mean(validation_accuracies)
                val_acc /= val_total_size
                val_loss /= val_total_size
                print(f"validation accuracy: {val_acc}, loss: {val_loss}")
                f1 = self.report_f1_precision_recall('validation')
                self.print_confusion_matrix_results("Validation", val_acc)

                self.valid_logs.append((self.step_no, val_acc, val_loss.item()))
                self.experiment.log_metric('validation_accuracy', val_acc, step=self.step_no)
                self.experiment.log_metric('validation_loss', val_loss.item(), step=self.step_no)

                if self.eval_metric == 'f1':
                    self.update_eval_metric_and_save_model(f1)
                else:
                    self.update_eval_metric_and_save_model(val_acc)

        predictions = self.evaluate()
        self.dump_logs_and_predictions(predictions)

