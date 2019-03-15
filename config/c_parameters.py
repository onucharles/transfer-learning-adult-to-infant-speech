# configuration parameters for classical models

c_parameters = {
        'mode': 'eval', # 'model_selection' or 'train_eval' or 'eval' or 'noisy_eval'
        'seed': 5,
        'log_experiment': True,

        # 'mode' == 'eval' or 'noisy_eval'
        'source_model': '/mnt/hdd/Experiments/chillanto-svm/704543ed6f50428d8242c7bf53a84ddf/train_eval.pkl',

        'svm_kernel': 'rbf', # 'polynomial' or 'rbf'

        # hyperparameter ranges. 'mode' == 'model_selection'
        'svm_hyperparam_range': {
            # np.logspace(-5, 1, 5)
            'gamma': [1.00000000e-05, 3.16227766e-04, 1.00000000e-02, 3.16227766e-01,     1.00000000e+01],
            # np.logspace(-2, 4, 5)
            'C': [1.00000000e-02, 3.16227766e-01, 1.00000000e+01, 3.16227766e+02, 1.00000000e+04]
        },

        'svm_no_folds': 5,

        # paths. 'log_experiment' == True
        'output_folder': '/mnt/hdd/Experiments/chillanto-svm',
        'grid_search_file': 'gridsearch.pkl',
        'train_eval_file': 'train_eval.pkl',

        # training setting. 'mode' == 'train_eval'
        'svm_train_params': {
            'C': 10,
            'gamma': 1e-5
        },

        # SpeechDataset params
        "group_speakers_by_id":  True,
        "silence_prob":  0.0,
        "noise_prob":  0.0,
        "input_length":  8000,
        "timeshift_ms": 100,
        "unknown_prob" :0.0,
        "train_pct": 80,
        "dev_pct": 5,
        "test_pct": 40,
        "wanted_words": ["normal", "asphyxia"], # ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"],
        "data_folder": "/mnt/hdd/Datasets/chillanto-8k-16bit-renamed",
        "sampling_freq": 8000,
        "n_dct_filters": 40,
        "n_mels": 40,
        "window_size_ms": 30,
        "frame_shift_ms": 10,
        "cache_size": 32768,
}