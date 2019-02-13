import numpy as np
import torch
import torch.nn as nn
import torch
from pathlib import Path
from sklearn.externals import joblib
from src.training_helpers import print_eval, set_seed, compute_eval, confusion_matrix, print_f1_confusion_matrix


class TrainerAndEvaluator():
    def __init__(self, task_params):
        self.experiment = task_params['experiment']
        config = task_params['config']
        self.train_loader, self.dev_loader, self.test_loader = task_params['loaders']
        self.device = task_params['device']

        set_seed(config)
        self.setup_model(task_params)
        self.setup_optimizers(config)


        self.step_no = 0
        self.max_acc = 0
        self.train_logs = []
        self.valid_logs = []

        self.n_epochs = config['n_epochs']
        self.schedule = config['schedule']
        self.n_labels = config['n_labels']
        self.dev_every = config['dev_every']
        self.print_confusion_matrix = config['print_confusion_matrix']

        self.setup_paths(config, self.experiment)
        self.experiment.log_parameters(config)

    def setup_model(self, task_params):
        model = task_params['model']
        model.to(self.device)
        self.model = torch.nn.DataParallel(model)

    def build_optimizer(self, config, lr):
        return torch.optim.SGD(self.model.parameters(), lr=lr, nesterov=config["use_nesterov"], weight_decay=config["weight_decay"], momentum=config["momentum"])

    def setup_optimizers(self, config):
        self.optimizers = [ self.build_optimizer(config, lr) for lr in config['lr'] ]
        self.optimizers.reverse()
        self.optimizer = self.optimizers.pop()

    def setup_paths(self, config, experiment):
        self.model_path = f"{config['model_path']}_{experiment.id}"
        self.predictions_path = f"{config['predictions_path']}_{experiment.id}.pred"
        self.log_file_path = f"{config['log_file_path']}_{experiment.id}.log"

    def report_training(self, train_score, loss):
        print_eval("train step #{0} {1}".format(self.step_no, "accuracy"), train_score, loss, self.optimizer.defaults['lr'])
        self.train_logs.append((self.step_no, train_score.item(), loss.item()))
        self.experiment.log_metric('train_loss', loss.item())
        self.experiment.log_metric('lr', self.optimizer.defaults['lr'])

    def report_validation(self, acc, loss):
        self.experiment.log_metric('dev_loss', loss.item())
        self.experiment.log_metric('dev_avg_acc', acc)

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
            torch.save(self.model.state_dict(), f"{self.model_path}_{self.max_acc}.mdl")

    def dump_logs_and_predictions(self, predictions):
        joblib.dump((self.train_logs, self.valid_logs), self.log_file_path)
        print('Training logs were written to: ', self.log_file_path)
        joblib.dump(predictions, self.predictions_path)

    def update_optimizer(self):
        if self.step_no in self.schedule:
            self.optimizer = self.optimizers.pop()
            print("changing learning rate to {}".format(self.optimizer))

    def evaluate(self):
        print("Evaluate")
        classes = np.arange(self.n_labels)
        criterion = nn.CrossEntropyLoss()
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
        acc = sum(results) / len(self.test_loader)

        print("Testing Accuracy", acc)
        self.report_confusion_matrix("Testing", acc)
        return prediction_log

    def perform_training(self):

        criterion = nn.CrossEntropyLoss()
        for epoch_idx in range(self.n_epochs):
            print(f"Epoch {epoch_idx}")
            for batch_idx, (model_in, labels) in enumerate(self.train_loader):
                self.step_no += 1
                self.model.train()
                self.optimizer.zero_grad()
                model_in = model_in.to(self.device)
                labels = labels.to(self.device)
                scores = self.model(model_in.clone().detach())

                loss = criterion(scores, labels)
                loss.backward()

                self.optimizer.step()
                self.update_optimizer()

                train_score = compute_eval(scores, labels)
                self.report_training(train_score, loss)

            if epoch_idx % self.dev_every == self.dev_every - 1:
                print("Validation")
                self.model.eval()
                accs = []
                self.reset_confusion_matrix()
                with torch.no_grad():
                    for model_in, labels in self.dev_loader:
                        model_in = model_in.to(self.device)
                        labels = labels.to(self.device)
                        scores = self.model(model_in)
                        loss = criterion(scores, labels)
                        loss_numeric = loss.item()
                        accs.append(compute_eval(scores, labels).detach().cpu())
                        self.update_confusion_matrix(scores, labels)

                avg_acc = np.mean(accs)
                self.report_confusion_matrix("Validation", avg_acc)

                self.valid_logs.append((self.step_no, avg_acc, loss.item()))
                self.report_validation(avg_acc, loss)
                self.update_accuracy_and_save_model(avg_acc)

        predictions = self.evaluate()
        self.dump_logs_and_predictions(predictions)

