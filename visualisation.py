""" Utility functions for visualising data and results """

import tensorflow as tf

def print_set_stats(audio_processor):
    """Prints the distribution of training, validation and test sets across the classes
    """
    data_index = audio_processor.data_index
    index_stats = {'validation': {}, 'testing': {}, 'training': {}}
    for set_index in ['training', 'validation', 'testing']:
        data_list = data_index[set_index]
        for example in data_list:
            cur_class = example['label']
            if cur_class in index_stats[set_index]:
                index_stats[set_index][cur_class] += 1
            else:
                index_stats[set_index][cur_class] = 1
    tf.logging.info('Training set contains: %s', index_stats['training'])
    tf.logging.info('Validation set contains: %s', index_stats['validation'])
    tf.logging.info('Testing set contains: %s', index_stats['testing'])