""" Tests a model on a given test data.

Sample script:
python test.py \
--model_checkpoint='/mnt/hdd/Experiments/chillanto8k/20181011-082349/train_checkpoints/conv.ckpt-1400' \
--model_parameters='/mnt/hdd/Experiments/chillanto8k/20181011-082349/parameters.json'

"""
import argparse

import input_data
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin

from baseline import models
from utils.ioutils import load_json
from visualisation import print_set_stats


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint to be evaluated.')
    parser.add_argument(
        '--model_parameters',
        type=str,
        required=True,
        help='Path to json file which contains parameters used to train model.')
    parser.add_argument(
        '--set',
        type=str,
        default='testing',
        choices=['training', 'validation', 'testing'],
        help='Which set to evaluate using specified checkpoint')

    FLAGS, unparsed = parser.parse_known_args()
    MODEL_FLAGS = load_flags(FLAGS.model_parameters)
    return FLAGS, MODEL_FLAGS

def load_flags(json_path):
    """ Load parameters from json file into """
    
    param_dict = load_json(json_path)
    flags = argparse.Namespace(**param_dict)
    return flags

def main():
    FLAGS, MODEL_FLAGS = parse_input()

    tf.logging.set_verbosity(tf.logging.INFO)

    # Start a new TensorFlow session.
    sess = tf.InteractiveSession()

    # prepare model settings
    model_settings = models.prepare_model_settings(
        len(input_data.prepare_words_list(MODEL_FLAGS.wanted_words.split(','))),
        MODEL_FLAGS.sample_rate, MODEL_FLAGS.clip_duration_ms, MODEL_FLAGS.window_size_ms,
        MODEL_FLAGS.window_stride_ms, MODEL_FLAGS.feature_bin_count, MODEL_FLAGS.preprocess)

    # create audio processor which manages loading, partitioning and preparing of audio data
    audio_processor = input_data.AudioProcessor(
        MODEL_FLAGS.data_url, MODEL_FLAGS.data_dir,
        MODEL_FLAGS.silence_percentage, MODEL_FLAGS.unknown_percentage,
        MODEL_FLAGS.wanted_words.split(','), MODEL_FLAGS.validation_percentage,
        MODEL_FLAGS.testing_percentage, model_settings, MODEL_FLAGS.summaries_dir)
    print_set_stats(audio_processor)

    # create model based on settings
    fingerprint_size = model_settings['fingerprint_size']
    fingerprint_input = tf.placeholder(tf.float32, [None, fingerprint_size], name='fingerprint_input')

    logits, dropout_prob = models.create_model(
        fingerprint_input,
        model_settings,
        MODEL_FLAGS.model_architecture,
        is_training=True
    )

    # get predictions and evaluation metrics
    predicted_indices = tf.argmax(logits, 1)
    label_count = model_settings['label_count']
    ground_truth_input = tf.placeholder(tf.int64, [None], name='groundtruth_input')
    correct_prediction = tf.equal(predicted_indices, ground_truth_input)
    confusion_matrix = tf.confusion_matrix(
        ground_truth_input, predicted_indices, num_classes=label_count)
    accuracy_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    recall_step, recall_op = tf.metrics.recall(ground_truth_input, predicted_indices)
    precision_step, precision_op = tf.metrics.precision(ground_truth_input, predicted_indices)

    # TP = tf.count_nonzero(predicted_indices * ground_truth_input, dtype=tf.float32)
    # TN = tf.count_nonzero((predicted_indices - 1) * (ground_truth_input - 1), dtype=tf.float32)
    # FP = tf.count_nonzero(predicted_indices * (ground_truth_input - 1), dtype=tf.float32)
    # FN = tf.count_nonzero((predicted_indices - 1) * ground_truth_input, dtype=tf.float32)
    # precision_step = TP / (TP + FP)
    # recall_step = TP / (TP + FN)
    # f1 = 2 * precision_step * recall_step / (precision_step + recall_step)

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()  # we need this because precision and recall have local variables.

    # load checkpoint
    if FLAGS.model_checkpoint:
        models.load_variables_from_checkpoint(sess, FLAGS.model_checkpoint)

    # testing
    set_size = audio_processor.set_size('testing')
    tf.logging.info('set_size=%d', set_size)
    total_accuracy = 0
    total_conf_matrix = None
    total_precision = 0
    total_recall = 0
    for i in xrange(0, set_size, MODEL_FLAGS.batch_size):
        test_fingerprints, test_ground_truth = audio_processor.get_data(
            MODEL_FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
        test_accuracy, conf_matrix, test_precision, test_recall = sess.run(
            [accuracy_step, confusion_matrix, precision_step, recall_step],
            feed_dict={
                fingerprint_input: test_fingerprints,
                ground_truth_input: test_ground_truth,
                dropout_prob: 1.0
            })
        batch_size = min(MODEL_FLAGS.batch_size, set_size - i)
        total_accuracy += (test_accuracy * batch_size) / set_size
        total_precision += (test_precision * batch_size) / set_size
        test_recall += (test_recall * batch_size) / set_size
        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix
    tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
    tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100,
                                                             set_size))
    tf.logging.info('Final test precision = %.1f%% (N=%d)' % (total_precision * 100,
                                                             set_size))
    tf.logging.info('Final test recall = %.1f%% (N=%d)' % (total_recall * 100,
                                                             set_size))

    # TODO add precision, recall and specificity (and F1?)
    # need to add these to the graph, so that we can optimise for each as we wish.
    # problem is the tf.precision function which class does it use?

if __name__=='__main__':
    main()