"""
Utilities for training classical models.

"""

from sklearn.metrics import recall_score, make_scorer, precision_score, f1_score, accuracy_score
import time
from sklearn.model_selection import StratifiedKFold, GridSearchCV

def train_eval(X_train, y_train, X_test, y_test, pipeline):

    # train classifier
    pipeline.fit(X_train, y_train)

    # evaluate
    y_pred_proba = pipeline.predict_proba(X_test)
    y_pred = pipeline.predict(X_test)

    return pipeline, y_test, y_pred, y_pred_proba

def model_select_skfold(X_train, y_train, pipeline, parameters, n_splits=5, random_state=42,
                        cache_size=4000, n_jobs=-1):
    """

    :param X_train:
    :param y_train:
    :param pipeline:
    :param parameters:
    :param n_splits:
    :param random_state:
    :param cache_size:
    :param n_jobs:
    :return:
    """

    #important settings
    refit = 'acc'
    fail_label = 1
    succ_label = 0

    scoring = {'acc': make_scorer(accuracy_score)}
    # scoring = {'bal_acc': make_scorer(recall_score, pos_label=fail_label, average='macro', greater_is_better=True),
    #            'success_rate': make_scorer(evalutils.success_rate, success_label=succ_label, greater_is_better=True),
    #            'failure_rate': make_scorer(evalutils.failure_rate, failure_label=fail_label, greater_is_better=True),
    #            'precision_failure': make_scorer(precision_score, pos_label=fail_label, greater_is_better=True),
    #            'f1_score_failure': make_scorer(f1_score, pos_label=fail_label, greater_is_better=True)}

    start = time.time()
    cv_gen = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(pipeline, parameters, cv=cv_gen, refit=refit, scoring=scoring, return_train_score=True,
                               verbose=4, n_jobs=n_jobs)
    grid_search.fit(X_train, y_train)
    end = time.time()
    print('Grid search complete. Took: {0} minutes'.format((end - start) / 60.0))

    return grid_search

def make_hyperparams(clf_name, hyperparmeters_dict):
    """
    Append classifier name to hyperparameters and return hyperparameters dictionary.
    :param clf_name: The name assigned to the classifier to be appended to the hyperparameter names
                    (enables association of hyperparameter with the classifier in sklearn's GridSearch)
    :param hyperparmeters_dict:
    :return:
    """

    hyperparams = {}
    for param, value in hyperparmeters_dict.items():
        new_param = clf_name + '__' + param
        hyperparams[new_param] = value

    return hyperparams