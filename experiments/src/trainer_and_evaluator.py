import numpy as np
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
from pathlib import Path
from sklearn.externals import joblib
from src.training_helpers import print_eval, set_seed, compute_eval, confusion_matrix, print_f1_confusion_matrix


class TrainerAndEvaluator():
    def __init__(self, task_params):
        self.experiment = task_params['experiment']
        config = task_params['config']
        self.train_loader, self.dev_loader, self.test_loader = task_params['loaders']
        self.device = task_params['device']

        self.setup_model(task_params)

        self.epoch_no = None
        self.step_no = None
        self.max_acc = None

        self.n_epochs = config['n_epochs']
        self.schedule = config['schedule']
        self.n_labels = config['n_labels']
        self.dev_every = config['dev_every']
        self.print_confusion_matrix = config['print_confusion_matrix']

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
        self.model = torch.nn.DataParallel(model)
        #self.model = model

    def setup_paths(self, config, experiment):
        self.model_path = f"{config['model_path']}_{experiment.id}"
        self.predictions_path = f"{config['predictions_path']}_{experiment.id}.pred"
        self.log_file_path = f"{config['log_file_path']}_{experiment.id}.log"

    def report_training(self, train_score, loss):
        print("Training - epoch:{} step:{} accuracy:{} loss:{} learning_rate:{}"
            .format(self.epoch_no, self.step_no, train_score, loss, self.optimizer.defaults['lr']))
        self.train_logs.append((self.step_no, train_score.item(), loss.item()))
        self.experiment.log_metric('train/loss', loss.item(), self.step_no)

    def report_confusion_matrix(self, label, acc):
        if self.print_confusion_matrix:
            print_f1_confusion_matrix(label, acc, self.conf_mat)

    def update_confusion_matrix(self, scores, labels):
        if self.print_confusion_matrix:
            self.conf_mat += confusion_matrix(scores.detach().cpu(), labels.detach().cpu(), np.arange(self.n_labels))

    def reset_confusion_matrix(self):
        if self.print_confusion_matrix:
            self.conf_mat = np.zeros((self.n_labels, self.n_labels))

    def update_accuracy_and_save_model(self, avg_acc):
        # save best model
        if avg_acc > self.max_acc:
            print("saving best model...")
            self.max_acc = avg_acc
            self.best_model = self.model
            torch.save(self.model.state_dict(), f"{self.model_path}_{self.max_acc}.mdl")

    def dump_logs_and_predictions(self, predictions):
        joblib.dump((self.train_logs, self.valid_logs), self.log_file_path)
        print('Training logs were written to: ', self.log_file_path)
        joblib.dump(predictions, self.predictions_path)

    def evaluate(self):
        print("Evaluate")
        criterion = self.loss_func
        results = []
        total = 0
        self.reset_confusion_matrix()
        prediction_log = []
        with torch.no_grad():
            for model_in, labels in self.test_loader:
                model_in = model_in.to(self.device)
                labels = labels.to(self.device)
                scores = self.best_model(model_in.clone().detach())
                loss = criterion(scores, labels)
                model_in_size = model_in.size(0)
                results.append(compute_eval(scores, labels) * model_in_size)
                self.update_confusion_matrix(scores, labels)
                total += model_in.cpu().size(0)
                prediction_log.append((scores, labels))
        acc = sum(results) / total

        print("Testing Accuracy", acc)
        self.experiment.log_metric('test/avg_acc', acc)
        self.report_confusion_matrix("Testing", acc)
        return prediction_log

    def perform_training(self):
        self.step_no = 0
        self.max_acc = 0

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
            self.epoch_no = epoch_idx + 1

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
                training_accuracies.append(train_score.detach().cpu())
                self.report_training(train_score, loss)
            avg_train_acc = np.mean(training_accuracies)
            self.experiment.log_metric('train/avg_acc', avg_train_acc, step=self.step_no)

            if epoch_idx % self.dev_every == self.dev_every - 1:
                print("Validation")
                self.model.eval()
                validation_accuracies= []
                self.reset_confusion_matrix()
                with torch.no_grad():
                    for model_in, labels in self.dev_loader:
                        model_in = model_in.to(self.device)
                        labels = labels.to(self.device)

                        model_in = Variable(model_in, requires_grad=False)
                        scores = self.model(model_in)
                        labels = Variable(labels, requires_grad=False)
                        loss = criterion(scores, labels)
                        valid_scores = compute_eval(scores, labels)
                        validation_accuracies.append(valid_scores.detach().cpu())
                        self.update_confusion_matrix(scores, labels)
                        print(f"validation accuracy. {valid_scores}, loss: {loss}")
                        self.experiment.log_metric('dev/loss', loss.item(), step=self.step_no)

                avg_dev_acc = np.mean(validation_accuracies)
                self.report_confusion_matrix("Validation", avg_dev_acc)

                self.valid_logs.append((self.step_no, avg_dev_acc, loss.item()))
                self.experiment.log_metric('dev/avg_acc', avg_dev_acc, step=self.step_no)
                self.update_accuracy_and_save_model(avg_dev_acc)

        predictions = self.evaluate()
        self.dump_logs_and_predictions(predictions)

