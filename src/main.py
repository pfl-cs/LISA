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

def bulk_loading(raw_data, temp_dir, model_dir_init):
    sorted_data_path = os.path.join(temp_dir, Config().static_data_name)
    params_path = os.path.join(temp_dir, Config().cell_params_path)
    preprocessing_flag = True
    if os.path.exists(params_path) == False or os.path.exists(sorted_data_path) == False:
        preprocessing_flag = False

    if preprocessing_flag == False:
        # one_dim_mappings_path = os.path.join(data_dir, 'original_one_dim_mappings.npy')
        # cell_measures_path = os.path.join(data_dir, 'cell_measures.npy')
        sorted_data, original_one_dim_mappings, params, cell_measures = layout_utils.generate_grid_cells(raw_data,
                                                                                                         Config().T_each_dim,
                                                                                                         Config().n_piecewise_models,
                                                                                                         Config().min_value,
                                                                                                         Config().max_value,
                                                                                                         Config().eta)

        np.save(sorted_data_path, sorted_data)
        np.save(params_path, params)
    else:
        params = np.load(params_path)

    my_idx = LISA(params=params, data_dim=Config().data_dim, page_size=Config().page_size, sigma=Config().sigma)
    my_idx.set_model_dir(model_dir_init)

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

        print 'n_models =', n_models
        start = 0
        # for i in range(n_models - 1, 0, -1):
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
            print i, 'finished'



    build_LISA_flag = my_idx.check_and_load_params()

    #----------------build LISA------------------
    if build_LISA_flag == False:
        if one_dim_mappings is None:
            one_dim_mappings = np.load(one_dim_mappings_path)
        my_idx.generate_pages(sorted_data, one_dim_mappings, col_split_idxes)
        my_idx.save()

    if one_dim_mappings is None:
        one_dim_mappings = np.load(one_dim_mappings_path)

    sorted_data = np.load(sorted_data_path)

    return my_idx, one_dim_mappings, sorted_data


def lattice_regression_preprocessing(my_idx, tau, lattice_data_dir):
    FileViewer.detect_and_create_dir(lattice_data_dir)

    lattice_nodes_path = os.path.join(lattice_data_dir, 'lattice_nodes.npy')
    nodes_radiuses_path = os.path.join(lattice_data_dir, 'nodes_radiuses.npy')
    lattice_training_points_path = os.path.join(lattice_data_dir, 'training_points.npy')
    lattice_training_radiuses_path = os.path.join(lattice_data_dir, 'training_radiuses.npy')
    knn_testing_points_path = os.path.join(lattice_data_dir, 'testing_points.npy')
    knn_testing_radiuses_path = os.path.join(lattice_data_dir, 'testing_radiuses.npy')

    paths = [lattice_nodes_path, nodes_radiuses_path, lattice_training_points_path,
             lattice_training_radiuses_path, knn_testing_points_path, knn_testing_radiuses_path]

    flag = False
    for path in paths:
        if os.path.exists(path) == False:
            flag = True
            break

    lattice_nodes = None
    lattice_training_points = None
    knn_testing_points = None
    if flag == True:
        if os.path.exists(lattice_nodes_path) and os.path.exists(lattice_training_points_path) and os.path.exists(knn_testing_points_path):
            lattice_nodes = np.load(lattice_nodes_path)
            lattice_training_points = np.load(lattice_training_points_path)
            knn_testing_points = np.load(knn_testing_points_path)
        else:
            lattice_nodes = my_idx.lattice_nodes_gen(tau)
            n = 40
            lattice_training_points = my_idx.sampling(lattice_nodes.shape[0] * n)
            knn_testing_points = my_idx.sampling(lattice_nodes.shape[0] * 10)

            np.save(lattice_nodes_path, lattice_nodes)
            np.save(lattice_training_points_path, lattice_training_points)
            np.save(knn_testing_points_path, knn_testing_points)

        if os.path.exists(nodes_radiuses_path) == False:
            nodes_radiuses = my_idx.get_estimate_radiuses(lattice_nodes, tau, 10)
            np.save(nodes_radiuses_path, nodes_radiuses)

        if os.path.exists(lattice_training_radiuses_path) == False:
            lattice_training_radiuses = my_idx.get_estimate_radiuses(lattice_training_points, tau, 10)
            np.save(lattice_training_radiuses_path, lattice_training_radiuses)

        if os.path.exists(knn_testing_radiuses_path) == False:
            knn_testing_radiuses = my_idx.get_estimate_radiuses(knn_testing_points, tau, 10)
            np.save(knn_testing_radiuses_path, knn_testing_radiuses)



def lattice_regression_train(my_idx, tau, lattice_data_dir, lattice_model_dir):
    lattice_regression_preprocessing(my_idx, tau, lattice_data_dir)
    FileViewer.detect_and_create_dir(lattice_model_dir)
    lat_reg = LatticeRegression()
    if lat_reg.if_need_train(lattice_model_dir) == True:
        lat_reg.train(lattice_data_dir)
        lat_reg.save(lattice_model_dir)



if __name__ == '__main__':

    Config()
    print 'home_dir =', Config().home_dir
    print 'data_dir =', Config().data_dir
    temp_dir = Config().data_dir
    raw_data = np.load(os.path.join(temp_dir, Config().static_data_name))
    model_dir_init = os.path.join(Config().models_dir, 'LISA_Init')
    my_idx, one_dim_mappings, sorted_data = bulk_loading(raw_data, temp_dir, model_dir_init)

    query_range_strs = FileViewer.load_list(Config().query_range_path)
    query_ranges = []
    for line in query_range_strs:
        query_ranges.append([float(i) for i in line.strip().split(' ')])
    query_ranges = np.array(query_ranges, dtype=np_data_type())
    # query_ranges = query_ranges[0:100]
    #
    # Fig 7 & Fig 11: --------------range query on init--------------------
    # my_idx = LISA()
    # my_idx.set_model_dir(model_dir_init)
    # flag = my_idx.check_and_load_params()
    # assert (flag == True)
    # total_n_pages, n_entries = my_idx.range_query(query_ranges)

    # Fig 8 & Fig 12: --------------range query on AI-----------------------
    # my_idx.set_model_dir(model_dir_init)
    # my_idx.check_and_load_params()
    # data_to_insert = np.load(os.path.join(temp_dir, Config().data_to_insert_name))
    # my_idx.insert(data_to_insert)
    # model_dir_AI = os.path.join(Config().models_dir, 'LISA_AI')
    # my_idx.set_model_dir(model_dir_AI)
    # my_idx.save()
    # total_n_pages, n_entries = my_idx.range_query(query_ranges)
    # print '#Pages =', total_n_pages

    # Fig 9: ---------------range query on AD-----------------------
    # my_idx.set_model_dir(model_dir_AI)
    # my_idx.check_and_load_params()
    # data_to_delete = np.load(os.path.join(temp_dir, Config().data_to_delete_name))
    # my_idx.delete(data_to_delete)
    # model_dir_AD = os.path.join(Config().models_dir, 'LISA_AD')
    # my_idx.set_model_dir(model_dir_AD)
    # my_idx.save()
    # total_n_pages, n_entries = my_idx.range_query(query_ranges)
    # print '#Pages =', total_n_pages

    # Fig 13: --------------KNN query--------------------
    my_idx = LISA()
    my_idx.set_model_dir(model_dir_init)
    my_idx.check_and_load_params()

    tau = Config().tau
    lattice_data_dir = os.path.join(Config().data_dir, 'lattice')
    lattice_data_dir = os.path.join(lattice_data_dir, str(tau))
    lattice_model_dir = os.path.join(Config().models_dir, 'lattice_regression')
    lattice_model_dir = os.path.join(lattice_model_dir, str(tau))

    lattice_regression_train(my_idx, tau, lattice_data_dir, lattice_model_dir)
    my_idx.load_knn_model(lattice_model_dir)

    knn_testing_points_path = os.path.join(lattice_data_dir, 'testing_points.npy')
    query_centers = np.load(knn_testing_points_path)
    query_centers = query_centers[0:1000]

    K = 3
    queried_keys = my_idx.knn_query(query_centers, K)
