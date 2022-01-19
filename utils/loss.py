import tensorflow as tf


def pi_loss(max_num_parts, model):
    params_tensor = tf.convert_to_tensor([model.get_layer(f'Projection{i}').trainable_weights[0] for i in range(max_num_parts)])
    # params_tensor.shape == (num_parts, encoding_dims, encoding_dims)
    pi_loss1 = tf.reduce_sum(tf.norm(params_tensor ** 2 - params_tensor, axis=[-2, -1]) ** 2)
    pi_loss2 = list()
    for idx, each_param in enumerate(params_tensor):
        unstack_params = tf.unstack(params_tensor, axis=0)
        del unstack_params[idx]
        new_params = tf.convert_to_tensor(unstack_params)
        pi_loss2.append(tf.norm(each_param * new_params, axis=[-2, -1]) ** 2)
    pi_loss2 = tf.reduce_sum(pi_loss2)
    pi_loss3 = tf.norm(tf.reduce_sum(params_tensor, axis=0) - tf.eye(tf.shape(params_tensor)[1])) ** 2
    return pi_loss1 + pi_loss2 + pi_loss3


def part_reconstruction_loss(gt, recon):
    return tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(gt, recon, from_logits=False), axis=(1, 2, 3, 4)))


def transformation_loss(gt, trans):
    return tf.nn.l2_loss(gt - trans) / tf.cast(tf.shape(gt)[0], dtype=tf.float32)


def shape_reconstruction_loss(gt, recon):
    return tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(gt, recon, from_logits=False), axis=(1, 2, 3)))
