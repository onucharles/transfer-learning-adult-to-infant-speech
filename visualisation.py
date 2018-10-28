from sklearn.externals import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse


FLAGS = None

def plot_learning_curve(train_log, valid_log, save_fig_path=None):
    plt.figure()

    # plot accuracy
    plt.subplot(121)
    plt.plot(train_log[:,0], train_log[:,1], label='training')
    plt.plot(valid_log[:,0], valid_log[:,1], label='validation')
    plt.xlabel('no of training steps')
    plt.ylabel('accuracy')
    plt.legend()

    # plot loss
    plt.subplot(122)
    plt.plot(train_log[:,0], train_log[:,2], label='training')
    plt.plot(valid_log[:,0], valid_log[:,2], label='validation')
    plt.xlabel('no of training steps')
    plt.ylabel('loss')
    plt.legend()

    if save_fig_path:
        plt.savefig(save_fig_path, bbox_inches='tight')
        print('Saving learning curve to: {0}'.format(save_fig_path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str)
    parser.add_argument('--save_fig_path', type=str)
    FLAGS, unparsed = parser.parse_known_args()

    # load files
    train_logs, valid_logs = joblib.load(FLAGS.log_file)

    # call function
    plot_learning_curve(np.array(train_logs), np.array(valid_logs), FLAGS.save_fig_path)    

if __name__ == "__main__":
    main()