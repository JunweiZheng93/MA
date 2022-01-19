import tensorflow as tf
import os
import numpy as np
import scipy.io
import argparse
import importlib
from utils import visualization


PROJ_ROOT = os.path.abspath(__file__)[:-15]
CATEGORY_MAP = {'chair': '03001627', 'table': '04379243', 'airplane': '02691156', 'lamp': '03636649'}


def evaluate_model(model_path,
                   mode='batch',
                   category='chair',
                   num_to_visualize=4,
                   single_shape_path=None,
                   visualize_decoded_part=False,
                   decoded_part_threshold=0.5,
                   transformed_part_threshold=0.5):

    hparam = importlib.import_module(f"results.{model_path.split('/')[-3]}.hparam")
    model = importlib.import_module(f"results.{model_path.split('/')[-3]}.model")

    if hparam.hparam['category'] == 'table':
        max_num_parts = 3
        whole_model = model.get_model(hparam.hparam['category'], max_num_parts, hparam.hparam['attention'], hparam.hparam['multi_inputs'])
    else:
        max_num_parts = 4
        whole_model = model.get_model(hparam.hparam['category'], max_num_parts, hparam.hparam['attention'], hparam.hparam['multi_inputs'])
    if hparam.hparam['process'] == 1 or hparam.hparam['process'] == '1':
        my_model = tf.keras.Model(whole_model.input, whole_model.get_layer('tf.stack').output, name=whole_model.name)
    elif hparam.hparam['process'] == 3 or hparam.hparam['process'] == '3':
        my_model = tf.keras.Model(whole_model.inputs, [whole_model.get_layer('Resampling').output, whole_model.get_layer('tf.stack').output], name=whole_model.name)
    else:
        raise ValueError(f'The model should be a model of process1 or process3. Got process{hparam.hparam["process"]} instead.')
    my_model.load_weights(model_path, by_name=True)

    if mode == 'batch':
        dataset_path = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category])
        all_shapes = os.listdir(dataset_path)
        idx = np.random.choice(len(all_shapes), num_to_visualize, replace=False)
        shapes_to_visualize = [all_shapes[each] for each in idx]

        for shape_code in shapes_to_visualize:
            # visualize ground truth label
            gt_label_path = os.path.join(dataset_path, shape_code, 'object_labeled.mat')
            gt_label = scipy.io.loadmat(gt_label_path)['data']
            gt_label = np.transpose(gt_label, (1, 0, 2))
            visualization.visualize(gt_label, title=shape_code)

            # visualize predicted label
            shape_path = os.path.dirname(gt_label_path)
            gt_shape_path = os.path.join(shape_path, 'object_unlabeled.mat')
            visualize_pred_label(my_model, 'batch', gt_shape_path, shape_code, visualize_decoded_part,
                                 decoded_part_threshold, transformed_part_threshold, hparam.hparam['process'])

    elif mode == 'single':
        # visualize ground truth label
        if not single_shape_path.endswith('/'):
            single_shape_path += '/'
        shape_code = single_shape_path.split('/')[-2]
        gt_label = scipy.io.loadmat(os.path.join(single_shape_path, 'object_labeled.mat'))['data']
        gt_label = np.transpose(gt_label, (1, 0, 2))
        visualization.visualize(gt_label, title=shape_code)

        # visualize predicted label
        gt_shape_path = os.path.join(single_shape_path, 'object_unlabeled.mat')
        visualize_pred_label(my_model, 'single', gt_shape_path, shape_code, visualize_decoded_part,
                             decoded_part_threshold, transformed_part_threshold, hparam.hparam['process'])


def visualize_pred_label(model,
                         mode,
                         shape_path,
                         shape_code,
                         visualize_decoded_part,
                         decoded_part_threshold,
                         transformed_part_threshold,
                         process):

    def get_pred_label(pred):
        code = 0
        for idx, each_part in enumerate(pred):
            code += each_part * 2 ** (idx + 1)
        pred_label = tf.math.floor(tf.experimental.numpy.log2(code + 1))
        pred_label = pred_label.numpy().astype('uint8')
        return pred_label

    gt_shape = tf.convert_to_tensor(scipy.io.loadmat(shape_path)['data'], dtype=tf.float32)
    gt_shape = tf.expand_dims(gt_shape, 0)
    gt_shape = tf.expand_dims(gt_shape, 4)
    if process == 1 or process == '1':
        parts = model(gt_shape)
        pred = tf.squeeze(tf.where(parts > decoded_part_threshold, 1., 0.))
        pred = pred.numpy().astype('uint8')
        for part in pred:
            part = np.transpose(part, (1, 0, 2))
            visualization.visualize(part, title=shape_code)
    elif process == 3 or process == '3':
        transformed_parts, parts = model(gt_shape)
        if visualize_decoded_part:
            pred = tf.squeeze(tf.where(parts > decoded_part_threshold, 1., 0.))
            pred = pred.numpy().astype('uint8')
            for part in pred:
                part = np.transpose(part, (1, 0, 2))
                visualization.visualize(part, title=shape_code)
        else:
            pred = tf.squeeze(tf.where(transformed_parts > transformed_part_threshold, 1., 0.))
            pred_label = get_pred_label(pred)
            visualization.visualize(pred_label, title=shape_code)
    else:
        raise ValueError(f'The model should be a model of process1 or process3. Got process{process} instead.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='this is the script to visualize reconstructed shape')

    parser.add_argument('model_path', help='path of the model')
    parser.add_argument('-m', '--mode', default='batch', help='visualize a batch of shapes or just a single shape. '
                                                              'Valid values are batch, single, exchange and assembly. '
                                                              'Default is batch')
    parser.add_argument('-c', '--category', default='chair', help='which kind of shape to visualize. Default is chair')
    parser.add_argument('-n', '--num_to_visualize', default=4, help='the number of shape to be visualized. Only valid'
                                                                    'when \'mode\' is \'batch\'')
    parser.add_argument('-s', '--single_shape_path', default=None, help='path of the shape to be visualized. e.g. '
                                                                        'datasets/03001627/1a6f615e8b1b5ae4dbbc9440457e303e. Only '
                                                                        'valid when \'mode\' is \'single\'')
    parser.add_argument('-v', '--visualize_decoded_part', action="store_true",
                        help='whether to visualize decoded parts')
    parser.add_argument('-d', '--decoded_part_threshold', default=0.5,
                        help='threshold of decoded parts to be visualized. '
                             'Default is 0.5')
    parser.add_argument('-t', '--transformed_part_threshold', default=0.5,
                        help='threshold of transformed parts to be visualized,'
                             'Default is 0.5')
    args = parser.parse_args()

    evaluate_model(model_path=args.model_path,
                   mode=args.mode,
                   category=args.category,
                   num_to_visualize=int(args.num_to_visualize),
                   single_shape_path=args.single_shape_path,
                   visualize_decoded_part=args.visualize_decoded_part,
                   decoded_part_threshold=float(args.decoded_part_threshold),
                   transformed_part_threshold=float(args.transformed_part_threshold))
