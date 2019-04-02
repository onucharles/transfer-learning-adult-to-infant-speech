
import os
import numpy as np
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
from pathlib import Path
from sklearn.externals import joblib
from src.training_helpers import print_eval, set_seed, compute_eval, confusion_matrix, print_f1_confusion_matrix, calc_f1_prec_recall, calc_sens_spec_uar

from src.evaluator import Evaluator

class NoiseEvaluator(Evaluator):
    def __init__(self, task_params):
        super(NoiseEvaluator, self).__init__(task_params)
        config = task_params['config']
        self.noise_pct = config['noise_pct'] * 100

    def report_f1_precision_recall(self, label):
        if self.print_confusion_matrix:
            f1, precision, recall = calc_f1_prec_recall(self.conf_mat)
            self.experiment.log_metric(f'{label}_F1', f1, step=self.noise_pct)
            self.experiment.log_metric(f'{label}_precision', precision, step=self.noise_pct)
            self.experiment.log_metric(f'{label}_recall', recall, step=self.noise_pct)

            sens, spec, uar = calc_sens_spec_uar(self.conf_mat)
            self.experiment.log_metric(f'{label}_sensitivity', sens)
            self.experiment.log_metric(f'{label}_specificity', spec)
            self.experiment.log_metric(f'{label}_UAR', uar)
