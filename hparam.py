hparam = {
    # -------------- dataset relevant ---------------------

    'category': 'chair',  # one of chair, table, airplane and lamp
    'batch_size': 64,
    'split_ratio': 0.8,  # ratio to split dataset into training set and test set
    'use_dist': False,  # whether to use gaussian distribution for the missing transformation matrix

    # -------------- model relevant -----------------------

    'attention': False,  # whether to use attention layer in the model
    'multi_inputs': False,  # whether to use latent representation and decoded parts as inputs for attention layer
    'process': 1,  # should be one of 1, 2 and 3
    'model_path': '/Users/junweizheng/Desktop/MA/results/20220119170926/process2/checkpoint.h5',  # model path for process2 or process 3

    # ---------------- training relevant ------------------

    'epochs': 500,
    'optimizer': 'adam',  # adam or sgd
    'lr': 1e-3,
    'decay_rate': 0.8,  # decay rate of learning rate
    'decay_step_size': 200,

    # ---------------- other settings --------------------

    'which_gpu': 0,
}
