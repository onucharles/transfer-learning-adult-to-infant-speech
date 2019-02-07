import numpy as np
import torch
import torch.nn as nn
import torch
from sklearn.externals import joblib
from src.training_helpers import print_eval, set_seed, evaluate, compute_eval, confusion_matrix, print_f1_confusion_matrix


def build_optimizer(model, config, lr):
    return torch.optim.SGD(model.parameters(), lr=lr, nesterov=config["use_nesterov"], weight_decay=config["weight_decay"], momentum=config["momentum"])

def task_train_and_evaluate(task_params):
    experiment = task_params['experiment']
    config = task_params['config']
    train_loader, dev_loader, test_loader = task_params['loaders']
    model = task_params['model']
    device = task_params['device']

    set_seed(config)
    model.to(device)
    optimizers = [ build_optimizer(model, config, lr) for lr in config['lr'] ]
    optimizers.reverse()

    experiment.log_parameters(config)
    optimizer = optimizers.pop()
    criterion = nn.CrossEntropyLoss()
    step_no = 0
    max_acc = 0
    train_logs = []
    valid_logs = []
    for epoch_idx in range(config["n_epochs"]):
        for batch_idx, (model_in, labels) in enumerate(train_loader):
            step_no += 1

            model.train()
            optimizer.zero_grad()
            model_in = model_in.to(device)
            labels = labels.to(device)
            scores = model(model_in.clone().detach())
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            if step_no in config["schedule"]:
                optimizer = optimizers.pop()
                print("changing learning rate to {}".format(optimizer))

            train_score = compute_eval(scores, labels)
            print_eval("train step #{0} {1}".format(step_no, "accuracy"), train_score, loss, optimizer.defaults['lr'])
            train_logs.append((step_no, train_score.item(), loss.item()))
            experiment.log_metric('train_loss', loss.item())
            experiment.log_metric('lr', optimizer.defaults['lr'])

        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            n_labels = config["n_labels"]
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

            print_f1_confusion_matrix("Validation", avg_acc, conf_mat)
            valid_logs.append((step_no, avg_acc, loss.item()))
            experiment.log_metric('dev_loss', loss.item())
            experiment.log_metric('dev_avg_acc', avg_acc)

            # save best model
            if avg_acc > max_acc:
                print("saving best model...")
                max_acc = avg_acc
                torch.save(model.state_dict(), config['model_path'])


    joblib.dump((train_logs, valid_logs), config['log_file_path'])
    print('Training logs were written to: ', config['log_file_path'])
    # Test model
    predictions = evaluate(config['n_labels'], model, device, test_loader)
    joblib.dump(predictions, config['predictions_path'])
