import tensorflow as tf
import os
from datetime import datetime
from tensorflow.keras.utils import Progbar
from model import get_model
from hparam import hparam
from utils import dataloader, loss


PROJ_ROOT = os.path.abspath(__file__)[:-8]
RESULT_PATH = os.path.join(PROJ_ROOT, 'results', datetime.now().strftime("%Y%m%d%H%M%S"))


def configure_gpu(which_gpu):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[which_gpu], "GPU")


def save_scripts(result_saved_path):

    if not os.path.exists(result_saved_path):
        os.makedirs(result_saved_path)

    # save training notes
    ans = input('The training will start soon. Do you want to take some notes for the training? y/n: ')
    while True:
        if ans == 'Y' or ans == 'yes' or ans == 'y':
            os.system(f'vi {os.path.join(result_saved_path, "training_notes.txt")}')
            print(f'training_notes.txt is saved in {result_saved_path}')
            break
        elif ans == 'N' or ans == 'no' or ans == 'n':
            break
        else:
            ans = input('Please enter y/n: ')

    # save __init__.py
    result_init_saved_path = os.path.join(os.path.dirname(result_saved_path), '__init__.py')
    os.system(f'touch {result_init_saved_path}')
    experiment_init_saved_path = os.path.join(result_saved_path, '__init__.py')
    os.system(f'touch {experiment_init_saved_path}')

    # save hparam.py
    hparam_saved_path = os.path.join(result_saved_path, 'hparam.py')
    hparam_source_path = os.path.join(PROJ_ROOT, 'hparam.py')
    os.system(f'cp {hparam_source_path} {hparam_saved_path}')

    # save model.py
    model_saved_path = os.path.join(result_saved_path, 'model.py')
    model_source_path = os.path.join(PROJ_ROOT, 'model.py')
    os.system(f'cp {model_source_path} {model_saved_path}')

    # save train.py
    train_saved_path = os.path.join(result_saved_path, 'train.py')
    train_source_path = os.path.join(PROJ_ROOT, 'train.py')
    os.system(f'cp {train_source_path} {train_saved_path}')


def get_optimizer(opt, lr):
    if opt == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr)
    elif opt == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr)
    else:
        raise ValueError(f'{opt} is not valid!')
    return optimizer


def train_step_process1(x, y):
    with tf.GradientTape() as tape:
        output = process1_model(x, training=True)
        pi = pi_loss(y.shape[1], process1_model)
        part_recon = part_recon_loss(y, output)
        total = pi + part_recon
    grads = tape.gradient(total, process1_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, process1_model.trainable_weights))
    pi_loss_tracker.update_state(pi)
    part_reconstruction_loss_tracker.update_state(part_recon)
    total_loss_tracker.update_state(total)
    return pi, part_recon, total


def train_step_process2(x, y):
    with tf.GradientTape() as tape:
        output = process2_model(x, training=True)
        trans = trans_loss(y, output)
    grads = tape.gradient(trans, process2_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, process2_model.trainable_weights))
    trans_loss_tracker.update_state(trans)
    return trans


def train_step_process3(x, y0, y1):
    with tf.GradientTape() as tape:
        shape, part, tran = process3_model(x, training=True)
        pi = pi_loss(y0.shape[1], process3_model)
        part_recon = part_recon_loss(y0, part)
        trans = trans_loss(y1, tran)
        shape_recon = shape_recon_loss(tf.transpose(x, (0, 2, 1, 3, 4)), shape)
        total = pi + part_recon + trans + shape_recon
    grads = tape.gradient(total, process3_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, process3_model.trainable_weights))
    pi_loss_tracker.update_state(pi)
    part_reconstruction_loss_tracker.update_state(part_recon)
    trans_loss_tracker.update_state(trans)
    shape_reconstruction_loss_tracker.update_state(shape_recon)
    total_loss_tracker.update_state(total)
    return pi, part_recon, trans, shape_recon, total


if __name__ == '__main__':

    # disable warning and info message, only enable error message
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    configure_gpu(hparam['which_gpu'])
    save_scripts(RESULT_PATH)

    if hparam['category'] == 'table':
        max_num_parts = 3
    else:
        max_num_parts = 4

    if hparam['process'] == 1 or hparam['process'] == '1':
        # load and pre-process dataset
        training_set, test_set = dataloader.get_dataset(hparam['category'], max_num_parts, hparam['split_ratio'], hparam['batch_size'])
        # set some paths
        model_saved_path = os.path.join(RESULT_PATH, 'process1')
        tb_saved_path = os.path.join(RESULT_PATH, 'process1', 'logs', 'train')
        if not os.path.exists(tb_saved_path):
            os.makedirs(tb_saved_path)
        # get model
        whole_model = get_model(hparam['category'], max_num_parts, hparam['attention'], hparam['multi_inputs'])
        process1_model = tf.keras.Model(whole_model.input, whole_model.get_layer('tf.stack').output, name=whole_model.name)
        del whole_model
        # get loss
        part_recon_loss = loss.part_reconstruction_loss
        pi_loss = loss.pi_loss
        # create loss tracker
        pi_loss_tracker = tf.keras.metrics.Mean()
        part_reconstruction_loss_tracker = tf.keras.metrics.Mean()
        total_loss_tracker = tf.keras.metrics.Mean()
        # get optimizer
        optimizer = get_optimizer(hparam['optimizer'], hparam['lr'])
        # create summary writer
        summary_writer = tf.summary.create_file_writer(tb_saved_path)
        # start epoch
        for epoch in range(hparam['epochs']):
            print('epoch {}/{}'.format(epoch+1, hparam['epochs']))
            # reset trackers
            pi_loss_tracker.reset_state()
            part_reconstruction_loss_tracker.reset_state()
            total_loss_tracker.reset_state()
            # create progress bar
            num_steps = training_set.cardinality().numpy()
            pb = Progbar(num_steps)
            # lr decay
            if (epoch + 1) % hparam['decay_step_size'] == 0:
                optimizer.lr = optimizer.lr * (1 - hparam['decay_rate'])
            # start batch
            for i, (voxel_grids, parts, trans) in enumerate(training_set):
                pi, part_recon, total = train_step_process1(voxel_grids, parts)
                pb.update(current=i+1, values=[('pi', pi), ('part', part_recon), ('total', total)])
            # save the best model
            current_epoch_result = total_loss_tracker.result()
            if epoch >= 1:
                if current_epoch_result < last_epoch_result:
                    process1_model.save(os.path.join(model_saved_path, 'checkpoint.h5'))
            last_epoch_result = current_epoch_result
            # log result to tensorboard
            with summary_writer.as_default():
                tf.summary.scalar('pi loss', pi_loss_tracker.result(), step=epoch)
                tf.summary.scalar('part recon loss', part_reconstruction_loss_tracker.result(), step=epoch)
                tf.summary.scalar('total loss', total_loss_tracker.result(), step=epoch)

    elif hparam['process'] == 2 or hparam['process'] == '2':
        # load and pre-process dataset
        training_set, test_set = dataloader.get_dataset(hparam['category'], max_num_parts, hparam['split_ratio'], hparam['batch_size'], 2, hparam['use_dist'], RESULT_PATH)
        # set some paths
        model_saved_path = os.path.join(RESULT_PATH, 'process2')
        tb_saved_path = os.path.join(RESULT_PATH, 'process2', 'logs', 'train')
        if not os.path.exists(tb_saved_path):
            os.makedirs(tb_saved_path)
        # get model
        whole_model = get_model(hparam['category'], max_num_parts, hparam['attention'], hparam['multi_inputs'])
        process1_model = tf.keras.Model(whole_model.input, whole_model.get_layer('tf.stack').output, name=whole_model.name)
        if hparam['attention']:
            if hparam['multi_inputs']:
                process2_model = tf.keras.Model(whole_model.input, whole_model.get_layer('tf.reshape_1').output, name=whole_model.name)
            else:
                process2_model = tf.keras.Model(whole_model.input, whole_model.get_layer('tf.reshape').output, name=whole_model.name)
        else:
            process2_model = tf.keras.Model(whole_model.input, whole_model.get_layer('LocalizationNet').output, name=whole_model.name)
        process2_model.load_weights(hparam['model_path'], by_name=True)
        del whole_model
        # get loss
        trans_loss = loss.transformation_loss
        # create loss tracker
        trans_loss_tracker = tf.keras.metrics.Mean()
        # get optimizer
        optimizer = get_optimizer(hparam['optimizer'], hparam['lr'])
        # create summary writer
        summary_writer = tf.summary.create_file_writer(tb_saved_path)
        # start epoch
        for epoch in range(hparam['epochs']):
            print('epoch {}/{}'.format(epoch+1, hparam['epochs']))
            # freeze some layers
            for layer in process1_model.layers:
                if hparam['attention']:
                    if hparam['multi_inputs']:
                        process2_model.get_layer(layer.name).trainable = False
                    else:
                        if layer.name == 'PartDecoder' or layer.name == 'tf.stack':
                            continue
                        else:
                            process2_model.get_layer(layer.name).trainable = False
                else:
                    process2_model.get_layer(layer.name).trainable = False
            # reset trackers
            trans_loss_tracker.reset_state()
            # create progress bar
            num_steps = training_set.cardinality().numpy()
            pb = Progbar(num_steps)
            # lr decay
            if (epoch + 1) % hparam['decay_step_size'] == 0:
                optimizer.lr = optimizer.lr * (1 - hparam['decay_rate'])
            # start batch
            for i, (voxel_grids, parts, trans) in enumerate(training_set):
                transformation = train_step_process2(voxel_grids, trans)
                pb.update(current=i+1, values=[('trans', transformation)])
            # save the best model
            current_epoch_result = trans_loss_tracker.result()
            if epoch >= 1:
                if current_epoch_result < last_epoch_result:
                    for layer in process2_model.layers:
                        layer.trainable = True
                    process2_model.save(os.path.join(model_saved_path, 'checkpoint.h5'))
            last_epoch_result = current_epoch_result
            # log result to tensorboard
            with summary_writer.as_default():
                tf.summary.scalar('transformation loss', trans_loss_tracker.result(), step=epoch)

    elif hparam['process'] == 3 or hparam['process'] == '3':
        # load and pre-process dataset
        training_set, test_set = dataloader.get_dataset(hparam['category'], max_num_parts, hparam['split_ratio'], hparam['batch_size'], 3, hparam['use_dist'], loaded_path=hparam['model_path'])
        # set some paths
        model_saved_path = os.path.join(RESULT_PATH, 'process3')
        tb_saved_path = os.path.join(RESULT_PATH, 'process3', 'logs', 'train')
        if not os.path.exists(tb_saved_path):
            os.makedirs(tb_saved_path)
        # get model
        whole_model = get_model(hparam['category'], max_num_parts, hparam['attention'], hparam['multi_inputs'])
        if hparam['attention']:
            if hparam['multi_inputs']:
                process3_model = tf.keras.Model(whole_model.inputs, [whole_model.output, whole_model.get_layer('tf.stack').output, whole_model.get_layer('tf.reshape_1').output], name=whole_model.name)
            else:
                process3_model = tf.keras.Model(whole_model.inputs, [whole_model.output, whole_model.get_layer('tf.stack').output, whole_model.get_layer('tf.reshape').output], name=whole_model.name)
        else:
            process3_model = tf.keras.Model(whole_model.inputs, [whole_model.output, whole_model.get_layer('tf.stack').output, whole_model.get_layer('LocalizationNet').output], name=whole_model.name)
        process3_model.load_weights(hparam['model_path'], by_name=True)
        del whole_model
        # get loss
        pi_loss = loss.pi_loss
        part_recon_loss = loss.part_reconstruction_loss
        trans_loss = loss.transformation_loss
        shape_recon_loss = loss.shape_reconstruction_loss
        # create loss tracker
        pi_loss_tracker = tf.keras.metrics.Mean()
        part_reconstruction_loss_tracker = tf.keras.metrics.Mean()
        trans_loss_tracker = tf.keras.metrics.Mean()
        shape_reconstruction_loss_tracker = tf.keras.metrics.Mean()
        total_loss_tracker = tf.keras.metrics.Mean()
        # get optimizer
        optimizer = get_optimizer(hparam['optimizer'], hparam['lr'])
        # create summary writer
        summary_writer = tf.summary.create_file_writer(tb_saved_path)
        # start epoch
        for epoch in range(hparam['epochs']):
            print('epoch {}/{}'.format(epoch+1, hparam['epochs']))
            # reset trackers
            pi_loss_tracker.reset_state()
            part_reconstruction_loss_tracker.reset_state()
            trans_loss_tracker.reset_state()
            shape_reconstruction_loss_tracker.reset_state()
            total_loss_tracker.reset_state()
            # create progress bar
            num_steps = training_set.cardinality().numpy()
            pb = Progbar(num_steps)
            # lr decay
            if (epoch + 1) % hparam['decay_step_size'] == 0:
                optimizer.lr = optimizer.lr * (1 - hparam['decay_rate'])
            # start batch
            for i, (voxel_grids, parts, trans) in enumerate(training_set):
                pi, part_recon, transformation, shape_recon, total = train_step_process3(voxel_grids, parts, trans)
                pb.update(current=i+1, values=[('pi', pi), ('part', part_recon), ('trans', transformation), ('shape', shape_recon), ('total', total)])
            # save the best model
            current_epoch_result = total_loss_tracker.result()
            if epoch >= 1:
                if current_epoch_result < last_epoch_result:
                    process3_model.save(os.path.join(model_saved_path, 'checkpoint.h5'))
            last_epoch_result = current_epoch_result
            # log result to tensorboard
            with summary_writer.as_default():
                tf.summary.scalar('pi loss', pi_loss_tracker.result(), step=epoch)
                tf.summary.scalar('part recon loss', part_reconstruction_loss_tracker.result(), step=epoch)
                tf.summary.scalar('transformation loss', trans_loss_tracker.result(), step=epoch)
                tf.summary.scalar('shape recon loss', shape_reconstruction_loss_tracker.result(), step=epoch)
                tf.summary.scalar('total loss', total_loss_tracker.result(), step=epoch)

    else:
        raise ValueError('process should be one of 1, 2 and 3')
