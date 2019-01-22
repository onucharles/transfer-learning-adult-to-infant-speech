from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from ConfigBuilder import ConfigBuilder
from config.c_parameters import c_parameters
import model as mod
from sklearn.manifold import TSNE
from sklearn.externals import joblib

def run_pca(data, labels, n_components=100):
    # calculate pca
    pca = PCA(n_components=n_components, whiten=True)
    data_new = pca.fit_transform(data)

    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    print('sum of variance is: ', cumulative_var[-1])

    # plot cumulative variance
    plt.figure()
    plt.plot(np.arange(len(cumulative_var)) + 1, cumulative_var)
    plt.grid()

    # plot first 2 principal components.
    data_normal = data_new[labels == 2, :]
    data_asphyxia = data_new[labels == 3, :]
    plt.figure()
    plt.scatter(data_normal[:, 0], data_normal[:, 1], label='normal')
    plt.scatter(data_asphyxia[:, 0], data_asphyxia[:, 1], label='asphyxia')
    plt.legend()

    print('Top 2 principal components account for {0} variance'.format(np.sum(pca.explained_variance_ratio_[:2])))

    return data_new

def run_tsne(data, labels):
    data_embedded = TSNE(n_components=2).fit_transform(data)

    data_normal = data_embedded[labels == 2, :]
    data_asphyxia = data_embedded[labels == 3, :]

    plt.figure()
    plt.scatter(data_normal[:,0], data_normal[:,1], label='normal')
    plt.scatter(data_asphyxia[:,0], data_asphyxia[:,1], label='asphyxia')
    plt.legend()

    joblib.dump(data_embedded, '/mnt/hdd/Dropbox (NRP)/paper/tsne_data/mfcc_tsne_test_only.pkl')

def main():
    # load parameters
    builder = ConfigBuilder(
        c_parameters,
    )
    parser = builder.build_argparse()
    config = builder.config_from_argparse(parser)

    # PLOTTING PCA
    train_set, dev_set, test_set = mod.SpeechDataset.splits(config)
    print('Loaded {0} training\t{1} validation\t{2} test examples'.format(len(train_set), len(dev_set), len(test_set)))
    # train_data, train_labels = mod.SpeechDataset.convert_dataset(train_set)
    # valid_data, valid_labels = mod.SpeechDataset.convert_dataset(dev_set)
    test_data, test_labels = mod.SpeechDataset.convert_dataset(test_set)
    # data = np.concatenate([train_data, valid_data, test_data])
    # labels = np.concatenate([train_labels, valid_labels, test_labels])
    # print('Merged training, validation and test sets. Data dimension is {0} and labels {1}'
    #       .format(data.shape, labels.shape))

    # run pca for raw mfcc features
    data_pca = run_pca(test_data, test_labels)

    # run pca for saved features
    embedding_path = '/mnt/hdd/Dropbox (NRP)/paper/tsne_data/output_embeddings_transfer.pkl'
    embeddings, labels = joblib.load(embedding_path)
    print('loaded embedding shape is {0} and labels shape is {1}'.format(embeddings.shape, labels.shape))
    data_pca = run_pca(embeddings, labels, n_components=None)

    plt.show()

if __name__ == "__main__":
    main()