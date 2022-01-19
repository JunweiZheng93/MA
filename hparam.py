hparam = {
    # -------------- dataset relevant ---------------------

    'category': 'chair',  # one of chair, table, airplane and lamp
    'batch_size': 2,
    'split_ratio': 0.8,  # ratio to split dataset into training set and test set
    'use_dist': False,  # whether to use gaussian distribution for the missing transformation matrix

    # -------------- model relevant -----------------------

    'attention': True,
    'multi_inputs': True,
    'process': 3,  # should be one of 1, 2 and 3
    'model_path': '/Users/junweizheng/Desktop/MA/results/20220119170926/process2/checkpoint.h5',

    # ---------------- training relevant ------------------

    'epochs': 3,
    'optimizer': 'adam',  # adam or sgd
    'lr': 1e-3,
    'decay_rate': 0.8,  # decay rate of learning rate
    'decay_step_size': 7,

    # ---------------- other settings --------------------

    'which_gpu': 0,
}
