import sys
import os

sys.path.append('../')

import numpy as np
from src.utils import FileViewer
from src.utils.core.config import Config
from src.utils import layout_utils
from src.utils import np_utils
from src.solution.LISA import LISA
from src.solution.piecewise_linear_curve_fit import PiecewiseModel
from src.FLAGS_DEFINE import *
import math
from src.solution.lattice_regression import LatticeRegression

def query_ranges_gen(start_low, start_high, offset_low, offset_high, max_value, dim, n):
    qr_list = []
    for _ in range(n):
        x = np.random.uniform(low=start_low, high=start_high, size=[dim*2])
        offset = np.random.uniform(low=offset_low, high=offset_high, size=[dim])
        y = (x[0:dim]+offset).clip(min=start_low,max=max_value)
        x[dim:] = y
        qr_list.append(x)

    query_ranges = np.array(qr_list)
    return query_ranges

def bulk_loading(raw_data, temp_dir):
    sorted_data_path = os.path.join(temp_dir, Config().static_data_name)
    params_path = os.path.join(temp_dir, Config().cell_params_path)
    preprocessing_flag = True
    if os.path.exists(params_path) == False or os.path.exists(sorted_data_path) == False:
        preprocessing_flag = False

    if preprocessing_flag == False:
        # one_dim_mappings_path = os.path.join(data_dir, 'original_one_dim_mappings.npy')
        # cell_measures_path = os.path.join(data_dir, 'cell_measures.npy')
        sorted_data, original_one_dim_mappings, params, cell_measures = layout_utils.generate_grid_cells(raw_data,
                                                                                                         Config().n_parts_each_dim,
                                                                                                         Config().n_piecewise_models,
                                                                                                         Config().min_value,
                                                                                                         Config().max_value,
                                                                                                         Config().eta)

        np.save(sorted_data_path, sorted_data)
        np.save(params_path, params)
    else:
        params = np.load(params_path)

    my_idx = LISA(params=params, data_dim=Config().data_dim, page_size=Config().page_size, sigma=Config().sigma)

    one_dim_mappings_path = os.path.join(Config().data_dir, 'one_dim_mappings.npy')
    col_split_idxes_path = os.path.join(Config().data_dir, 'col_split_idxes.npy')
    one_dim_mappings = None
    sorted_data = np.load(sorted_data_path)
    if os.path.exists(one_dim_mappings_path) == False or os.path.exists(col_split_idxes_path) == False:
        one_dim_mappings, col_split_idxes = my_idx.monotone_mappings_and_col_split_idxes_for_sorted_data(sorted_data)
        np.save(one_dim_mappings_path, one_dim_mappings)
        np.save(col_split_idxes_path, col_split_idxes)
    else:
        col_split_idxes = np.load(col_split_idxes_path)


    n_models = col_split_idxes.shape[0]
    piecewise_params_dir = os.path.join(Config().models_dir, 'piecewise')
    piecewise_linear_fit_flag = my_idx.check_and_load_piecewise_models_params(piecewise_params_dir, n_models)

    # ------------------piecewise linear curve fit-------------------
    if piecewise_linear_fit_flag == False:
        if one_dim_mappings is None:
            one_dim_mappings = np.load(one_dim_mappings_path)
        sigma = Config().sigma
        FileViewer.detect_and_create_dir(piecewise_params_dir)

        start = 0
        for i in range(n_models):
            if i > 0:
                start = col_split_idxes[i - 1]
            end = col_split_idxes[i]
            one_dim_input = one_dim_mappings[start:end]
            pm = PiecewiseModel(i, one_dim_input, sigma)
            model_dir = os.path.join(piecewise_params_dir, str(i))
            if os.path.exists(model_dir) == False:
                pm.train()
                FileViewer.detect_and_create_dir(model_dir)
                pm.save(model_dir)


    build_LISA_flag = my_idx.check_and_load_params()

    #----------------build LISA------------------
    if build_LISA_flag == False:
        if one_dim_mappings is None:
            one_dim_mappings = np.load(one_dim_mappings_path)
        my_idx.generate_pages(sorted_data, one_dim_mappings, col_split_idxes)
        # my_idx.set_model_dir(LISA_dir)
        my_idx.save()

    if one_dim_mappings is None:
        one_dim_mappings = np.load(one_dim_mappings_path)

    sorted_data = np.load(sorted_data_path)

    return my_idx, one_dim_mappings, sorted_data


def save_lattice_info(data_dir, lattice_nodes, nodes_radiuses, lattice_training_points, lattice_training_radiuses,
                      n_nodes_each_dim):
    lattice_dir = os.path.join(data_dir, 'lattice')
    lattice_dir = os.path.join(lattice_dir, str(n_nodes_each_dim))
    FileViewer.detect_and_create_dir(lattice_dir)
    lattice_nodes_path = os.path.join(lattice_dir, 'lattice_nodes.npy')
    nodes_radiuses_path = os.path.join(lattice_dir, 'nodes_radiuses.npy')
    lattice_training_points_path = os.path.join(lattice_dir, 'training_points.npy')
    lattice_training_radiuses_path = os.path.join(lattice_dir, 'training_radiuses.npy')
    np.save(lattice_nodes_path, lattice_nodes)
    np.save(nodes_radiuses_path, nodes_radiuses)
    np.save(lattice_training_points_path, lattice_training_points)
    np.save(lattice_training_radiuses_path, lattice_training_radiuses)


def lattice_regression_preprocessing(my_idx):
    n_nodes_each_dim = Config().n_nodes_each_dim
    lattice_data_dir = os.path.join(Config().data_dir, 'lattice')
    lattice_data_dir = os.path.join(lattice_data_dir, str(n_nodes_each_dim))
    FileViewer.detect_and_create_dir(lattice_data_dir)

    lattice_nodes_path = os.path.join(lattice_data_dir, 'lattice_nodes.npy')
    nodes_radiuses_path = os.path.join(lattice_data_dir, 'nodes_radiuses.npy')
    lattice_training_points_path = os.path.join(lattice_data_dir, 'training_points.npy')
    lattice_training_radiuses_path = os.path.join(lattice_data_dir, 'training_radiuses.npy')

    knn_testing_points_path = os.path.join(lattice_data_dir, 'testing_points.npy')

    lattice_nodes = my_idx.lattice_nodes_gen(n_nodes_each_dim)
    lattice_training_points = my_idx.lattice_training_data_gen(lattice_nodes.shape[0] * 40)
    knn_testing_points = my_idx.lattice_training_data_gen(lattice_nodes.shape[0] * 10)

    np.save(lattice_nodes_path, lattice_nodes)
    np.save(lattice_training_points_path, lattice_training_points)
    np.save(knn_testing_points_path, knn_testing_points)

    nodes_radiuses = my_idx.get_estimate_radiuses(lattice_nodes, n_nodes_each_dim, 20)
    lattice_training_radiuses = my_idx.get_estimate_radiuses(lattice_training_points, n_nodes_each_dim, 20)
    np.save(nodes_radiuses_path, nodes_radiuses)
    np.save(lattice_training_radiuses_path, lattice_training_radiuses)

    # my_idx.save_lattice_info(Config().data_dir, lattice_nodes, nodes_radiuses, lattice_training_points, lattice_training_radiuses, n_nodes_each_dim)

    n_lattices_each_dim = Config().n_nodes_each_dim
    lattice_model_dir = os.path.join(Config().models_dir, 'lattice_regression')
    lattice_model_dir = os.path.join(lattice_model_dir, str(n_lattices_each_dim))
    FileViewer.detect_and_create_dir(lattice_model_dir)
    lat_reg = LatticeRegression()
    lat_reg.train(lattice_data_dir)
    lat_reg.save(lattice_model_dir)


if __name__ == '__main__':

    workspace = '~/workspace/LISA/4d_uniform'
    Config(workspace)
    print 'home_dir =', Config().home_dir
    print 'data_dir =', Config().data_dir
    temp_dir = Config().data_dir
    raw_data = np.load(os.path.join(temp_dir, 'data_0.npy'))
    my_idx, one_dim_mappings, sorted_data = bulk_loading(raw_data, temp_dir)
    # total_pages = my_idx.query_single_thread(query_ranges)

    # --------------range query--------------------
    query_range_path = os.path.join(Config().data_dir, Config().query_range_path)
    query_range_strs = FileViewer.load_list(query_range_path)
    query_ranges = []
    for line in query_range_strs:
        query_ranges.append([float(i) for i in line.strip().split(' ')])
    query_ranges = np.array(query_ranges, dtype=np_data_type())
    n_pages, n_entries = my_idx.range_query(query_ranges)

    # --------------KNN query--------------------

    # lattice_model_dir = os.path.join(Config().models_dir, 'lattice_regression')
    # lattice_model_dir = os.path.join(lattice_model_dir, str(Config().n_nodes_each_dim))
    # my_idx.load_knn_model(lattice_model_dir)
    #
    #
    # lattice_data_dir = os.path.join(Config().data_dir, 'lattice')
    # lattice_data_dir = os.path.join(lattice_data_dir, str(Config().n_nodes_each_dim))
    # knn_testing_points_path = os.path.join(lattice_data_dir, 'testing_points.npy')
    # query_centers = np.load(knn_testing_points_path)
    # K = 10
    # all_queried_keys, total_n_pages, radiuses, init_radiuses, node_indices_list, n_pages_every_query = my_idx.knn_query(
    #     query_centers, K)

    # ---------------insert-----------------------
    # data_to_insert = np.load(os.path.join(temp_dir, 'data_1.npy'))
    # my_idx.insert(data_to_insert)

    # ---------------delete-----------------------
    # data_to_delete = np.load(os.path.join(temp_dir, 'data_2.npy'))
    # my_idx.delete(data_to_delete)




