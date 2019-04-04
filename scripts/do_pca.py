from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
# from ConfigBuilder import ConfigBuilder
# from config.c_parameters import c_parameters
# import model as mod

def run_pca(embedding_names, embedding_path_list):

    plt.figure(figsize=(6.5,4.5))
    for emb_name, emb_path in zip(embedding_names, embedding_path_list):
        data, labels = joblib.load(emb_path)
        print('loaded embedding shape is {0} and labels shape is {1}'.format(data.shape, labels.shape))

        # calculate pca
        pca = PCA(n_components=None, whiten=True)
        data_new = pca.fit_transform(data)

        cumulative_var = np.cumsum(pca.explained_variance_ratio_)
        top2_var = np.sum(pca.explained_variance_ratio_[:2])
        print('sum of variance is: ', cumulative_var[-1])
        print('Top 2 principal components account for {0} variance'.format(top2_var))
        top2_var_rounded = np.round(top2_var, 2)
        print(top2_var_rounded)

        # plot cumulative variance
        plt.plot(np.arange(len(cumulative_var)) + 1, cumulative_var, label=emb_name)
        plt.plot(2, top2_var, 'o', color='g')
        plt.text(2 + 2, top2_var, str(top2_var_rounded), fontsize=12)
    plt.xlabel('no of principal components', fontsize=13)
    plt.ylabel('% cumulative variance explained', fontsize=13)
    # plt.title('Cumulative variance explained')
    # plt.grid()
    plt.legend(fontsize=13)

    plt.figure(figsize=(6.5,4.5))
    for emb_name, emb_path in zip(embedding_names, embedding_path_list):
        data, labels = joblib.load(emb_path)
        print('loaded embedding shape is {0} and labels shape is {1}'.format(data.shape, labels.shape))

        # take average of embeddings.
        data_avg = np.mean(data, axis=0)

        # plot cumulative variance
        plt.plot(range(len(data_avg)), data_avg, label=emb_name)
    plt.xlabel('features', fontsize=13)
    plt.ylabel('weight', fontsize=13)
    plt.legend(fontsize=13)

    plt.show()

def main():
    # no-transfer
    no_embedding_path = '/mnt/hdd/Experiments/chillanto-pt-2/chill_embeddings/41d4cfb8f1fa4aa5a48ad2fb2ef554d7/output_embeddings_test.pkl'
    # sc-transfer
    sc_embedding_path = '/mnt/hdd/Experiments/chillanto-pt-2/chill_embeddings/00c72a94a6984632b5c0f5ac2a9b7578/output_embeddings_test.pkl'
    # vctk-transfer
    vctk_embedding_path = '/mnt/hdd/Experiments/chillanto-pt-2/chill_embeddings/02c48bc09ccd4144a64a2aa7678ef518/output_embeddings_test.pkl'
    # sitw-transfer
    sitw_embedding_path = '/mnt/hdd/Experiments/chillanto-pt-2/chill_embeddings/1ebcb55eb5e44e4f9393cb15e1a63480/output_embeddings_test.pkl'
    embedding_names = ['no-transfer', 'sc-transfer', 'vctk-transfer', 'sitw-transfer']
    embedding_path_list = [no_embedding_path, sc_embedding_path, vctk_embedding_path, sitw_embedding_path]
    run_pca(embedding_names, embedding_path_list)

# def run_pca(data, labels, n_components=None, title=None):
#     # calculate pca
#     pca = PCA(n_components=n_components, whiten=True)
#     data_new = pca.fit_transform(data)
#
#     cumulative_var = np.cumsum(pca.explained_variance_ratio_)
#     top2_var = np.sum(pca.explained_variance_ratio_[:2])
#     print('sum of variance is: ', cumulative_var[-1])
#     print('Top 2 principal components account for {0} variance'.format(top2_var))
#     top2_var_rounded = np.round(top2_var, 2)
#     print(top2_var_rounded)
#
#     # plot cumulative variance
#     plt.subplot(1,2,1)
#     plt.plot(np.arange(len(cumulative_var)) + 1, cumulative_var)
#     plt.plot(2, top2_var, 'o', color='g')
#     plt.text(2 + 1, top2_var, 'Top 2 (cum. variance=' + str(top2_var_rounded) + ')')
#     plt.xlabel('no principal components')
#     plt.ylabel('% cumulative variance explained')
#     # plt.title('Cumulative variance explained')
#     plt.grid()
#
#     # plot first 2 principal components.
#     plt.subplot(1,2,2)
#     data_normal = data_new[labels == 2, :]
#     data_asphyxia = data_new[labels == 3, :]
#     plt.scatter(data_normal[:, 0], data_normal[:, 1], label='normal')
#     plt.scatter(data_asphyxia[:, 0], data_asphyxia[:, 1], label='asphyxia')
#     plt.xlabel('first principal component')
#     plt.ylabel('second principal component')
#     # plt.title('Top 2 principal components (variance=' + str(top2_var_rounded) + ')')
#     plt.legend()
#
#     return data_new
#
# def main():
#     # load parameters
#     builder = ConfigBuilder(
#         c_parameters,
#     )
#     parser = builder.build_argparse()
#     config = builder.config_from_argparse(parser)
#
#     # PLOTTING PCA
#     train_set, dev_set, test_set = mod.SpeechDataset.splits(config)
#     print('Loaded {0} training\t{1} validation\t{2} test examples'.format(len(train_set), len(dev_set), len(test_set)))
#     # train_data, train_labels = mod.SpeechDataset.convert_dataset(train_set)
#     # valid_data, valid_labels = mod.SpeechDataset.convert_dataset(dev_set)
#     test_data, test_labels = mod.SpeechDataset.convert_dataset(test_set)
#     # data = np.concatenate([train_data, valid_data, test_data])
#     # labels = np.concatenate([train_labels, valid_labels, test_labels])
#     # print('Merged training, validation and test sets. Data dimension is {0} and labels {1}'
#     #       .format(data.shape, labels.shape))
#
#     # run pca for raw mfcc features
#     # data_pca = run_pca(test_data, test_labels)
#
#     # run pca for res8 no-transfer
#     embedding_path = '/mnt/hdd/Experiments/chillanto-pt/20190308-142245/output_embeddings_test.pkl'
#     embeddings, labels = joblib.load(embedding_path)
#     print('loaded embedding shape is {0} and labels shape is {1}'.format(embeddings.shape, labels.shape))
#     data_pca = run_pca(embeddings, labels, n_components=None)
#
#     # # run pca for res8 transfer
#     # embedding_path = '/mnt/hdd/Experiments/chillanto-pt/20190308-141918/output_embeddings_train.pkl'
#     # embeddings, labels = joblib.load(embedding_path)
#     # print('loaded embedding shape is {0} and labels shape is {1}'.format(embeddings.shape, labels.shape))
#     # data_pca = run_pca(embeddings, labels, n_components=None)
#
#
#     plt.show()

if __name__ == "__main__":
    main()