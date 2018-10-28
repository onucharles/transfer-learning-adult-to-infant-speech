from sklearn.externals import joblib
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def print_grid_search_result(grid_search, sort_by='mean_test_acc', num_results=10, df_names_dict={}):
    """
    Prints the result of a grid search (using scikit-learn GridSearchCV module) to the screen.
    :param grid_search: object of type 'GridSearchCV'
    :return:
    """

    print('Best score = {0}'.format(round(grid_search.best_score_, 2)))
    print('Best params = {0}'.format(grid_search.best_params_))

    best_idx = grid_search.best_index_
    print('On test data...')
    print('\taccuracy = {0}'.format(round(grid_search.best_score_, 2)))

    # print train scores
    print('On train data...')
    print('\taccuracy = {0}'.format(round(grid_search.cv_results_['mean_train_acc'][best_idx], 2)))

    # Print top 10 results.
    results_df = pd.DataFrame(grid_search.cv_results_).sort_values(by=sort_by, ascending=False).head(
        num_results)
    results_df.rename(inplace=True, columns=df_names_dict)
    return results_df[list(df_names_dict.values())]

def main():

    # PRINTING GRID SEARCH
    grid_search_file = '/mnt/hdd/Experiments/chillanto-svm/20181027-141910/gridsearch.pkl'
    grid_search = joblib.load(grid_search_file)
    print_grid_search_result(grid_search)



if __name__ == "__main__":
    main()