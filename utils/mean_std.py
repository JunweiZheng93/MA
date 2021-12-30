import os
import numpy as np
import scipy.io


PROJ_ROOT = os.path.abspath(__file__)[:-18]
CATEGORY_MAP = {'chair': '03001627', 'table': '04379243', 'airplane': '02691156', 'lamp': '03636649'}


def get_mean_std(category='chair'):
    """
    :return: it has shape of (num_parts, num_params, num_statistics). num_params always equal to 4, because we have 4x4
    parameters but actually only 4 different parameters in transformation matrix. num_statistics always equal to 2,
    because it stands for mean and std for the 4 different parameters in transformation matrix.
    """
    if category == 'table':
        max_num_parts = 3
    else:
        max_num_parts = 4
    category_path = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category])
    shape_path = [os.path.join(category_path, shape) for shape in os.listdir(category_path)]
    stat = list()
    for i in range(1, max_num_parts + 1):
        temp = list()
        for shape in shape_path:
            transformation_path = os.path.join(shape, f'part{i}_trans_matrix.mat')
            if os.path.exists(transformation_path):
                transformation = scipy.io.loadmat(transformation_path)['data']
                temp.append(transformation)
            else:
                continue
        mean = np.mean(temp, axis=0)
        std = np.std(temp, axis=0)
        stat.append(((mean[0, 0], std[0, 0]), (mean[0, 3], std[0, 3]), (mean[1, 3], std[1, 3]), (mean[2, 3], std[2, 3])))
    stat = np.array(stat, dtype='float32')
    return stat


if __name__ == '__main__':
    stat = get_mean_std('chair')
    print(stat)
