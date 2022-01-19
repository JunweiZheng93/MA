import os
import numpy as np
import tensorflow as tf
import math
import scipy.io

CATEGORY_MAP = {'chair': '03001627', 'table': '04379243', 'airplane': '02691156', 'lamp': '03636649'}
URL_MAP = {'chair': 'https://gitlab.com/JunweiZheng93/shapenetsegvox/-/raw/master/03001627.zip?inline=false',
           'table': 'https://gitlab.com/JunweiZheng93/shapenetsegvox/-/raw/master/04379243.zip?inline=false',
           'airplane': 'https://gitlab.com/JunweiZheng93/shapenetsegvox/-/raw/master/02691156.zip?inline=false',
           'lamp': 'https://gitlab.com/JunweiZheng93/shapenetsegvox/-/raw/master/03636649.zip?inline=false'}
PROJ_ROOT = os.path.abspath(__file__)[:-19]


def get_dataset(category='chair', max_num_parts=4, split_ratio=0.8, batch_size=32, process=1, use_dist=False, saved_path=None, loaded_path=None):

    category_path = download_dataset(category)

    voxel_grid_fp, part_fp, trans_fp = get_fp(category_path)
    num_training_samples = math.ceil(len(voxel_grid_fp) * split_ratio)
    print(f'total: {len(voxel_grid_fp)} samples - training set: {num_training_samples} samples - '
          f'test set: {len(voxel_grid_fp)-num_training_samples} samples - split ratio: {split_ratio}')

    all_voxel_grid = list()
    all_part = list()
    all_trans = list()
    stat = get_statistics(category)

    print('Loading data to memory. It will take a while. Please wait...')
    for v_fp, p_fp, t_fp in zip(voxel_grid_fp, part_fp, trans_fp):
        v = scipy.io.loadmat(v_fp)['data'][:, :, :, np.newaxis]
        all_voxel_grid.append(v)

        parts = list()
        transformations = list()
        member_list = [int(each[-5]) for each in p_fp]
        dir_name = os.path.dirname(v_fp)
        for i in range(1, max_num_parts + 1):
            if i not in member_list:
                part = np.zeros_like(v, dtype='uint8')
                parts.append(part)
                if process == 2 and use_dist:
                    scale = np.random.normal(stat[i-1, 0, 0], stat[i-1, 0, 1])
                    translation1 = np.random.normal(stat[i-1, 1, 0], stat[i-1, 1, 1])
                    translation2 = np.random.normal(stat[i-1, 2, 0], stat[i-1, 2, 1])
                    translation3 = np.random.normal(stat[i-1, 3, 0], stat[i-1, 3, 1])
                    transformation = np.zeros((3, 4), dtype='float32')
                    transformation[0, 0] = transformation[1, 1] = transformation[2, 2] = scale
                    transformation[0, 3] = translation1
                    transformation[1, 3] = translation2
                    transformation[2, 3] = translation3
                    path = os.path.join(saved_path, 'transformations', v_fp.split('/')[-2])
                    os.makedirs(path)
                    scipy.io.savemat(os.path.join(path, f'part{i}_trans_matrix.mat'), {'data': transformation})
                elif process == 3 and use_dist:
                    path = loaded_path[:-23]
                    transformation = scipy.io.loadmat(os.path.join(path, 'transformations', v_fp.split('/')[-2], f'part{i}_trans_matrix.mat'))['data']
                else:
                    transformation = np.zeros((3, 4), dtype='float32')
                transformations.append(transformation)
            else:
                part = scipy.io.loadmat(os.path.join(dir_name, f'part{i}.mat'))['data'][:, :, :, np.newaxis]
                parts.append(part)
                transformations.append(scipy.io.loadmat(os.path.join(dir_name, f'part{i}_trans_matrix.mat'))['data'][:3])
        all_part.append(parts)
        all_trans.append(transformations)

    training_set = tf.data.Dataset.from_tensor_slices((all_voxel_grid[:num_training_samples],
                                                       all_part[:num_training_samples],
                                                       all_trans[:num_training_samples]))
    training_set = training_set.shuffle(len(voxel_grid_fp), reshuffle_each_iteration=True)
    training_set = training_set.batch(batch_size, drop_remainder=False)
    training_set = training_set.prefetch(tf.data.AUTOTUNE)

    test_set = None
    return training_set, test_set


def download_dataset(category):

    # check category
    if category not in list(CATEGORY_MAP.keys()):
        raise ValueError(f'category should be one of chair, table, airplane and lamp. got {category} instead!')

    category_path = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category])
    if not os.path.exists(category_path):
        tf.keras.utils.get_file(f'{CATEGORY_MAP[category]}.zip', URL_MAP[category], cache_dir=PROJ_ROOT, extract=True)
        os.remove(f'{category_path}.zip')
    return category_path


def get_fp(category_fp):
    shape_paths = sorted([os.path.join(category_fp, shape_name) for shape_name in os.listdir(category_fp)])
    voxel_grid_fp = list()
    part_fp = list()
    trans_fp = list()
    for shape_path in shape_paths:
        voxel_grid = os.path.join(shape_path, 'object_unlabeled.mat')
        part_list = list()
        trans_list = list()
        all_files = sorted(os.listdir(shape_path))
        for file in all_files:
            if file.startswith('part') and file.endswith('.mat'):
                if file.startswith('part') and file.endswith('trans_matrix.mat'):
                    trans_list.append(os.path.join(shape_path, file))
                    continue
                part_list.append(os.path.join(shape_path, file))
        voxel_grid_fp.append(voxel_grid)
        part_fp.append(part_list)
        trans_fp.append(trans_list)
    return voxel_grid_fp, part_fp, trans_fp


def get_statistics(category='chair'):
    if category == 'chair':
        data = [[[1.58, 0.28], [0.71, 4.18], [-19.98, 8.42], [-9.47, 4.54]],
                [[1.74, 0.37], [-13.81, 5.95], [-9.44, 7.25], [-11.99, 5.95]],
                [[1.66, 0.64], [-11.46, 16.10], [3.23, 3.12], [-10.66, 13.86]],
                [[1.51, 1.45], [-10.32, 31.60], [-11.88, 24.14], [-7.99, 20.51]]]
    elif category == 'table':
        data = [[[1.01, 0.08], [-0.27, 1.41], [-7.53, 4.84], [-0.26, 1.40]],
                [[1.23, 1.32], [-3.97, 22.19], [-2.37, 13.64], [-3.54, 23.35]],
                [[5.90, 9.68], [-81.79, 184.61], [-105.03, 202.61], [-77.95, 208.07]]]
    elif category == 'airplane':
        data = [[[1.12, 0.23], [-3.20, 5.15], [-0.33, 4.07], [-2.24, 4.10]],
                [[1.13, 0.18], [-1.97, 2.83], [0.002, 2.99], [-2.15, 3.02]],
                [[2.88, 1.02], [2.99, 8.20], [-33.85, 16.91], [-30.50, 23.11]],
                [[2.51, 2.23], [-27.64, 31.67], [-17.23, 32.24], [-24.11, 35.00]]]
    else:
        data = [[[3.72, 2.37], [-36.68, 37.96], [7.55, 6.28], [-43.67, 38.05]],
                [[2.53, 1.47], [-30.51, 31.61], [-41.44, 48.86], [-24.73, 23.88]],
                [[6.30, 4.24], [-85.11, 69.06], [-170.42, 120.18], [-84.76, 67.59]],
                [[2.04, 3.52], [-16.65, 59.04], [-14.15, 47.67], [-16.72, 55.10]]]
    stat = np.array(data)
    return stat
