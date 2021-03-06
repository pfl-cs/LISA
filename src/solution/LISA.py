import sys
import os

sys.path.append('../../')

import numpy as np
from src.utils import FileViewer
from src.utils.core.config import Config
from src.utils import layout_utils
from src.utils import np_utils
from src.FLAGS_DEFINE import *
import time
import cPickle
from piecewise_linear_curve_fit import PiecewiseModel
import threading
import math
from lattice_regression import LatticeRegression


class LISA():
    def __init__(self, name='LISA', data_dim=2, params=None, page_size=-1, sigma=100):
        self.data_dim = data_dim
        self.name = name
        self.model_dir = os.path.join(Config().models_dir, self.name)
        # FileViewer.detect_and_create_dir(self.model_dir)

        # all params: data params and tree params
        self.page_size = page_size
        self.params = params
        if params is not None:
            self.params_dump()

        self.m_counts = []
        self.sigma = sigma

        self.col_split_shard_ids = None
        self.Alphas = None
        self.Betas = None

        self.n_threads = 32

    def set_model_dir(self, model_dir):
        self.model_dir = model_dir

    def setParams(self, params):
        self.params = params

    def setPageSize(self, page_size):
        self.page_size = page_size

    def params_dump(self):
        self.data_dim = int(self.params[-1])
        self.n_parts_each_dim = int(self.params[-2])
        self.n_piecewise_models = int(self.params[-3])
        self.max_value_each_dim = self.params[-4]
        self.min_value_each_dim = self.params[-5]
        self.eta = self.params[-6]

        self.all_split_upper_bounds = []
        self.all_front_split_points = []
        self.all_split_points_without_head_and_tail = []
        self.all_cell_lens = []

        start = 0
        n_repeat = 1
        for j in range(self.data_dim - 1):
            one_dim_split_upper_bounds = []
            for i in range(n_repeat):
                end = start + self.n_parts_each_dim
                split_upper_bounds = self.params[start:end]
                one_dim_split_upper_bounds.append(split_upper_bounds)
                start = end


            one_dim_split_upper_bounds = np.array(one_dim_split_upper_bounds,dtype=np_data_type())
            one_dim_front_split_points = np.zeros_like(one_dim_split_upper_bounds, dtype=one_dim_split_upper_bounds.dtype)
            one_dim_split_points_without_head_and_tail = one_dim_split_upper_bounds[:, 0:-1]
            one_dim_front_split_points[:, 1:] = one_dim_split_points_without_head_and_tail
            one_dim_cell_lens = one_dim_split_upper_bounds - one_dim_front_split_points

            self.all_split_upper_bounds.append(one_dim_split_upper_bounds)
            self.all_front_split_points.append(one_dim_front_split_points)
            self.all_split_points_without_head_and_tail.append(one_dim_split_points_without_head_and_tail)
            self.all_cell_lens.append(one_dim_cell_lens)
            n_repeat *= self.n_parts_each_dim


        self.borders, self.cell_measures = layout_utils.create_borders(self.all_split_upper_bounds)

        end = start + self.n_parts_each_dim
        self.last_dim_split_upper_bounds = np.array(self.params[start:end],dtype=np_data_type())
        self.last_dim_front_split_points = np.zeros_like(self.last_dim_split_upper_bounds, dtype=self.last_dim_split_upper_bounds.dtype)
        self.last_dim_split_points_without_tail = self.last_dim_split_upper_bounds[0:-1]
        self.last_dim_front_split_points[1:] = self.last_dim_split_points_without_tail
        self.last_dim_cell_lens = self.last_dim_split_upper_bounds - self.last_dim_front_split_points

        start = end
        end = start + self.n_piecewise_models
        self.model_split_mappings = self.params[start:end]
        self.model_split_mappings_without_tail = self.model_split_mappings[0:-1]
        assert (end == len(self.params) - 6)


    def cal_pred_idxes_in_particular_col(self, mappings, col_id, idx=-1, offset=-1):
        if idx < 0:
            alphas = self.Alphas[col_id]
            betas = self.Betas[col_id]
            A = PiecewiseModel.relu(np.tile(np.reshape(mappings, [-1, 1]), [1, self.sigma]) - betas.transpose())
            pred_idxes = A.dot(alphas)
            return pred_idxes
        else:
            alphas = self.Alphas[col_id]
            betas = self.Betas[col_id]
            A = PiecewiseModel.relu(np.tile(np.reshape(mappings, [-1, 1]), [1, self.sigma]) - betas.transpose())
            pred_idxes = A.dot(alphas)

            print '%%%col_id =', col_id
            print pred_idxes[idx] / self.page_size
            print pred_idxes[idx] / self.page_size + offset

            return pred_idxes

    @staticmethod
    def shards_layout(pred_idxes, N):
        pred_idxes = (pred_idxes / N).astype(np_idx_type())
        max_shard_id = pred_idxes.max()
        pred_idxes = np.clip(pred_idxes, a_min=0, a_max=max_shard_id)
        entries_count = [0] * (max_shard_id + 1)
        for i in range(pred_idxes.shape[0]):
            idx = pred_idxes[i]
            entries_count[idx] += 1

        n_entries_last_page = entries_count[max_shard_id]
        if n_entries_last_page < N:
            max_shard_id -= 1
            while True:
                n_entries_last_page += entries_count[max_shard_id]
                if n_entries_last_page > N:
                    break
                max_shard_id -= 1
            # max_shard_id += 1
            entries_count[max_shard_id] = n_entries_last_page
            entries_count = entries_count[0:max_shard_id + 1]
        return entries_count

    def generate_pages(self, sorted_data, one_dim_mappings, col_split_idxes):
        n_cols = col_split_idxes.shape[0]
        col_split_shard_ids = [0]
        self.shard_infos = []
        self.pages = []
        start = 0

        n_shards = 0
        page_no = 0

        col_min_mappings = []

        self.shard_ids_for_sorted_data = np.zeros(shape=[sorted_data.shape[0]], dtype=np_idx_type())
        self.col_ids_for_sorted_data = np.zeros(shape=[sorted_data.shape[0]], dtype=np_idx_type())
        shard_id = 0
        for i in range(n_cols):
            # print '------------i =', i, '--------------'
            end = col_split_idxes[i]
            self.col_ids_for_sorted_data[start:end] = i

            one_dim_input = one_dim_mappings[start:end]
            min_mapping = one_dim_input.min()
            col_min_mappings.append(min_mapping)

            pred_idxes = self.cal_pred_idxes_in_particular_col(one_dim_input - min_mapping, i)

            entries_count = self.shards_layout(pred_idxes, self.page_size)

            n_shards += len(entries_count)
            col_split_shard_ids.append(n_shards)
            print 'i =', i, 'n_shards =', n_shards, 'min_pred_idx =', pred_idxes.min(), 'n_shards_this_col =', len(
                entries_count)

            entry_start_idx = start
            for e_count in entries_count:
                shard_info = [[], []]
                entry_end_idx = entry_start_idx + e_count
                if e_count > 0:
                    self.shard_ids_for_sorted_data[entry_start_idx:entry_end_idx] = shard_id
                shard_id += 1
                if e_count > 0:
                    pages = sorted_data[entry_start_idx:entry_end_idx]
                    one_dim_pages = one_dim_mappings[entry_start_idx:entry_end_idx]
                    if e_count <= self.page_size:
                        # page_no = len(self.pages)
                        self.pages.append(pages)
                        shard_info[0].append(page_no)
                        page_no += 1
                    else:
                        n_pages = int(e_count / self.page_size)
                        if n_pages * self.page_size < e_count:
                            n_pages += 1

                        b_k = 0
                        for k in range(n_pages):
                            e_k = b_k + self.page_size
                            if e_k > e_count:
                                e_k = e_count
                            page = pages[b_k:e_k]
                            self.pages.append(page)
                            shard_info[0].append(page_no)
                            page_no += 1
                            if k > 0:
                                shard_info[1].append(one_dim_pages[b_k])
                            b_k = e_k

                self.shard_infos.append(shard_info)
                entry_start_idx = entry_end_idx

            start = end

        self.m_counts = []
        for page in self.pages:
            self.m_counts.append(page.shape[0])
        self.col_split_shard_ids = np.array(col_split_shard_ids, dtype=np_idx_type())
        self.cal_shard_numbers_each_col()
        self.col_min_mappings = np.array(col_min_mappings, dtype=np_data_type())

    def cal_shard_numbers_each_col(self):
        self.shard_numbers_each_col = np.zeros(shape=[self.col_split_shard_ids.shape[0] - 1], dtype=np_idx_type())
        for i in range(self.shard_numbers_each_col.shape[0]):
            self.shard_numbers_each_col[i] = self.col_split_shard_ids[i + 1] - self.col_split_shard_ids[i]


    def check_and_load_piecewise_models_params(self, piecewise_models_dir, n_models):
        self.Alphas = np.zeros(shape=[n_models, self.sigma], dtype=np.float64)
        self.Betas = np.zeros(shape=[n_models, self.sigma], dtype=np.float64)
        for i in range(n_models):
            model_dir = os.path.join(piecewise_models_dir, str(i))
            alphas_path = os.path.join(model_dir, 'alphas.npy')
            betas_path = os.path.join(model_dir, 'betas.npy')
            if os.path.exists(alphas_path) and os.path.exists(betas_path):
                self.Alphas[i] = np.load(alphas_path)
                self.Betas[i] = np.load(betas_path)
            else:
                return False
        return True


    def monotone_mappings(self, data):
        idxes = np.searchsorted(self.all_split_points_without_head_and_tail[0][0], data[:, 0], side='right')
        for i in range(1, data.shape[1] - 1):
            for j in range(idxes.shape[0]):
                idxes[j] = idxes[j] * self.n_parts_each_dim + np.searchsorted(self.all_split_points_without_head_and_tail[i][idxes[j]], data[j, i],side='right')

        last_dim_data = data[:, -1]
        left_data = data[:, 0:-1]

        measures = np.prod(left_data - self.borders[idxes], axis=1) / self.cell_measures[idxes]

        mappings = measures * self.eta + (last_dim_data / self.max_value_each_dim * (self.n_parts_each_dim - 1)) + (idxes * self.n_parts_each_dim)
        return mappings

    def monotone_mappings_and_col_split_idxes_for_sorted_data(self, sorted_points):
        mappings = self.monotone_mappings(sorted_points)
        col_idxes = np.searchsorted(self.model_split_mappings_without_tail, mappings, side='right')
        N = col_idxes[-1]
        print '----------------N =', N, col_idxes.max()
        col_split_idxes = [0] * (N + 1)
        for i in range(col_idxes.shape[0]):
            col_split_idxes[col_idxes[i]] += 1

        count = col_split_idxes[0]
        for i in range(1, N + 1):
            count += col_split_idxes[i]
            col_split_idxes[i] = count

        return mappings.astype(sorted_points.dtype), np.array(col_split_idxes)


    def get_intersect_shard(self, query_range, dim, cell_idx_base, lower_measure, upper_measure, lower_mappings, upper_mappings):
        lower_val = query_range[dim]
        upper_val = query_range[dim + self.data_dim]
        if dim == self.data_dim - 1:
            lower_measure *= self.eta
            upper_measure *= self.eta
            base = lower_measure + cell_idx_base * self.n_parts_each_dim
            lower_mapping = base + lower_val / self.max_value_each_dim * (self.n_parts_each_dim - 1)
            upper_mapping = base + upper_val / self.max_value_each_dim * (self.n_parts_each_dim - 1)
            lower_mappings.append(lower_mapping)
            upper_mappings.append(upper_mapping)

        else:
            split_points_without_head_and_tail = self.all_split_points_without_head_and_tail[dim][cell_idx_base]
            split_upper_bounds = self.all_split_upper_bounds[dim][cell_idx_base]
            front_split_points = self.all_front_split_points[dim][cell_idx_base]
            cell_lens = self.all_cell_lens[dim][cell_idx_base]
            lower_idx = np.searchsorted(split_points_without_head_and_tail, lower_val, side='right')
            upper_idx = np.searchsorted(split_points_without_head_and_tail, upper_val, side='right')

            if lower_idx == upper_idx:
                new_query_range = np.copy(query_range)
                new_lower_measure = lower_measure * (lower_val - front_split_points[lower_idx]) / cell_lens[lower_idx]
                new_upper_measure = upper_measure * (upper_val - front_split_points[upper_idx]) / cell_lens[upper_idx]
                self.get_intersect_shard(new_query_range, dim + 1, cell_idx_base * self.n_parts_each_dim + lower_idx,
                                         new_lower_measure, new_upper_measure, lower_mappings, upper_mappings)
            else:
                new_query_range = np.copy(query_range)
                new_query_range[dim + self.data_dim] = split_upper_bounds[lower_idx]
                new_lower_measure = lower_measure * (lower_val - front_split_points[lower_idx]) / cell_lens[lower_idx]
                self.get_intersect_shard(new_query_range, dim + 1, cell_idx_base * self.n_parts_each_dim + lower_idx,
                                         new_lower_measure, upper_measure, lower_mappings, upper_mappings)

                for next_one_dim_id in range(lower_idx + 1, upper_idx):
                    new_query_range = np.copy(query_range)
                    new_query_range[dim] = split_upper_bounds[next_one_dim_id - 1]
                    new_query_range[dim + self.data_dim] = split_upper_bounds[next_one_dim_id]
                    self.get_intersect_shard(new_query_range, dim + 1,
                                             cell_idx_base * self.n_parts_each_dim + next_one_dim_id,
                                             0, upper_measure, lower_mappings, upper_mappings)

                new_query_range = np.copy(query_range)
                new_query_range[dim] = split_upper_bounds[upper_idx - 1]
                new_upper_measure = upper_measure * (upper_val - front_split_points[upper_idx]) / cell_lens[upper_idx]
                self.get_intersect_shard(new_query_range, dim + 1, cell_idx_base * self.n_parts_each_dim + upper_idx,
                                         0, new_upper_measure, lower_mappings, upper_mappings)



    def get_query_ranges_mappings(self, query_ranges):
        num_query_points_each_query = []
        lower_mappings_list = []
        upper_mappings_list = []
        n_last = 0
        for i in range(query_ranges.shape[0]):
            query_range = query_ranges[i]
            single_qr_lower_mappings = []
            single_qr_upper_mappings = []
            self.get_intersect_shard(query_range, 0, 0, 1, 1, single_qr_lower_mappings, single_qr_upper_mappings)
            lower_mappings_list.extend(single_qr_lower_mappings)
            upper_mappings_list.extend(single_qr_upper_mappings)

            self.union_continuous_cells(single_qr_lower_mappings, single_qr_upper_mappings, lower_mappings_list, upper_mappings_list)

            num_query_points_each_query.append(len(upper_mappings_list) - n_last)
            n_last = len(upper_mappings_list)

        lower_mappings = np.array(lower_mappings_list, dtype=np_data_type())
        upper_mappings = np.array(upper_mappings_list, dtype=np_data_type())

        return lower_mappings, upper_mappings, num_query_points_each_query

    def union_continuous_cells(self, low_mappings, high_mappings, all_low_mappings, all_high_mappings):
        # n_cells = len(cell_ids)
        # for i in range(n_cells):
        #     all_low_mappings.append(low_mappings[i])
        #     all_high_mappings.append(high_mappings[i])
        n_cells = len(low_mappings)
        if n_cells > 0:
            # last_id = cell_ids[0]
            l_m = low_mappings[0]
            h_m = high_mappings[0]
            for i in range(1, n_cells):
                # curr_id = cell_ids[i]
                curr_low_mapping = low_mappings[i]
                assert (curr_low_mapping >= h_m)
                if curr_low_mapping - h_m < 1e-5:
                    h_m = high_mappings[i]
                else:
                    all_low_mappings.append(l_m)
                    all_high_mappings.append(h_m)
                    l_m = curr_low_mapping
                    h_m = high_mappings[i]
            all_low_mappings.append(l_m)
            all_high_mappings.append(h_m)

    @staticmethod
    def upper_bound(sorted_array, point):
        return np.clip((np.sign(point - sorted_array).astype(np.int64) + 1), a_min=0, a_max=1).sum()

    def predict_single_mapping_shard_id(self, mapping):
        col_idx = np.searchsorted(self.model_split_mappings_without_tail, mapping, side='right')
        # col_idx = int(mapping / self.max_column_measure)
        print 'col_idx =', col_idx
        trans_mapping = mapping - self.col_min_mappings[col_idx]
        shard_id_offset = self.col_split_shard_ids[col_idx]
        max_pred_idx = self.shard_numbers_each_col[col_idx] - 1
        print 'max_pred_idx =', max_pred_idx
        print 'shard_id_offset =', shard_id_offset
        print 'trans_mapping =', trans_mapping
        print self.col_split_shard_ids[col_idx + 1]

        alphas = self.Alphas[col_idx]
        betas = self.Betas[col_idx]

        pred_shard_id = min(max(int(np.sum(alphas * PiecewiseModel.relu(trans_mapping - betas)) / self.page_size), 0),
                           max_pred_idx)
        print 'pred_Shard_id =', pred_shard_id
        pred_shard_id += shard_id_offset
        return pred_shard_id

    def predict_shard_ids(self, mappings, print_flag=False):
        col_idxes = np.searchsorted(self.model_split_mappings_without_tail, mappings, side='right')
        # col_idxes = (mappings / self.max_column_measure).astype(np_idx_type()).clip(min=0,
        #                                                                             max=self.col_min_mappings.shape[
        #                                                                                     0] - 1)

        trans_mappings = mappings - self.col_min_mappings[col_idxes]

        shard_id_offsets = self.col_split_shard_ids[col_idxes]
        max_pred_idxes = self.shard_numbers_each_col[col_idxes] - 1

        all_alphas = self.Alphas[col_idxes]
        all_betas = self.Betas[col_idxes]

        all_A = PiecewiseModel.relu((trans_mappings - all_betas.transpose()).transpose())
        pred_shard_ids = (np.sum(all_A * all_alphas, axis=1) / self.page_size).astype(np_idx_type()).clip(min=0,
                                                                                                         max=max_pred_idxes)
        pred_shard_ids += shard_id_offsets

        return pred_shard_ids


    def predict_within_shard(self, mapping, shard_id, side):
        # print 'shard_id =', shard_id
        shard_info = self.shard_infos[shard_id]
        shard_split_mappings = shard_info[1]
        if len(shard_info[0]) == 0:
            return -1
        if len(shard_split_mappings) == 0:
            return 0
        else:
            idx = np.searchsorted(shard_split_mappings, mapping, side=side)
            return idx

    def get_query_page_nos(self, query_ranges):
        lower_mappings, upper_mappings, num_query_points_each_query = self.get_query_ranges_mappings(query_ranges)
        lower_shard_ids = self.predict_shard_ids(lower_mappings)
        upper_shard_ids = self.predict_shard_ids(upper_mappings)
        start = 0
        query_page_nos = []
        for i in range(len(num_query_points_each_query)):
            page_nos = []
            end = start + num_query_points_each_query[i]
            if end > start:
                query_lower_shard_ids = lower_shard_ids[start:end]
                query_upper_shard_ids = upper_shard_ids[start:end]
                query_lower_mappings = lower_mappings[start:end]
                query_upper_mappings = upper_mappings[start:end]
                for j in range(query_lower_shard_ids.shape[0]):
                    lower_shard_id = query_lower_shard_ids[j]
                    upper_shard_id = query_upper_shard_ids[j]

                    lower_shard_split_mappings = self.shard_infos[lower_shard_id][1]
                    upper_shard_split_mappings = self.shard_infos[upper_shard_id][1]
                    for _ in range(len(lower_shard_split_mappings)):
                        __ = lower_shard_split_mappings[_]

                    for _ in range(len(upper_shard_split_mappings)):
                        __ = upper_shard_split_mappings[_]

                    # shard_split_mappings
                    lower_idx = self.predict_within_shard(query_lower_mappings[j], lower_shard_id, 'left')
                    upper_idx = self.predict_within_shard(query_upper_mappings[j], upper_shard_id, 'right')
                    if lower_shard_id == upper_shard_id:
                        if lower_idx >= 0:
                            page_nos.extend(self.shard_infos[lower_shard_id][0][lower_idx:upper_idx + 1])
                    else:
                        if lower_idx >= 0:
                            page_nos.extend(self.shard_infos[lower_shard_id][0][lower_idx:])
                        for k in range(lower_shard_id + 1, upper_shard_id):
                            page_nos.extend(self.shard_infos[k][0])
                        if upper_idx >= 0:
                            page_nos.extend(self.shard_infos[upper_shard_id][0][0:upper_idx + 1])
                    # for k in range(lower_shard_id, upper_shard_id + 1):
                    #     page_nos.extend(self.shard_infos[k][0])
            page_nos = set(page_nos)
            query_page_nos.append(page_nos)

            start = end

        return query_page_nos

    def range_query(self, query_ranges):
        query_page_nos = self.get_query_page_nos(query_ranges)
        # print 'query_page_nos = ', query_page_nos
        n_entries_each_query = np.zeros(shape=[query_ranges.shape[0]], dtype=np_idx_type())
        n_pages_each_query = np.zeros(shape=[query_ranges.shape[0]], dtype=np_idx_type())

        for i in range(len(query_page_nos)):
            page_nos = query_page_nos[i]
            n_pages_each_query[i] = len(page_nos)
            query_results = []
            for page_no in page_nos:
                page = self.pages[page_no]
                query_results.append(page)

            if len(query_results) > 0:
                query_results = np.concatenate(query_results, axis=0)
                # query_results = query_results[self.overlap(query_results, query_ranges[i])]
                query_results = self.overlap(query_results, query_ranges[i])
                n_entries_each_query[i] = query_results.shape[0]

        return n_pages_each_query, n_entries_each_query

    def get_query_keys_within_sphericals(self, query_ranges, centers, radiuses):
        query_page_nos = self.get_query_page_nos(query_ranges)
        query_keys_list = []
        query_key_dists_list = []
        n_pages_list = []
        assert len(query_page_nos) == centers.shape[0]
        for i in range(len(query_page_nos)):
            # for i in range(1):
            page_nos = query_page_nos[i]
            n_pages_list.append(len(page_nos))
            query_results = []
            for page_no in page_nos:
                page = self.pages[page_no]
                query_results.append(page)

            if len(query_results) > 0:
                query_results = np.concatenate(query_results, axis=0)
                query_results, dists = self.overlap_spherical(query_results, centers[i], radiuses[i])
                query_keys_list.append(query_results)
                query_key_dists_list.append(dists)

        return query_keys_list, query_key_dists_list, n_pages_list

    def get_radius_for_knn_query(self, points, radius, K):
        query_ranges = np.zeros(shape=[points.shape[0], self.data_dim * 2], dtype=np_data_type())
        query_ranges[:, 0:self.data_dim] = points - radius
        query_ranges[:, self.data_dim:] = points + radius
        query_ranges = np.clip(query_ranges, a_min=Config().min_value, a_max=Config().max_value)

        radiuses = np.ones(shape=points.shape[0], dtype=points.dtype) * radius

        offset = 100000
        n_iters = int(query_ranges.shape[0] / offset)
        if offset * n_iters < query_ranges.shape[0]:
            n_iters += 1

        radius_list = [None] * points.shape[0]
        for i in range(n_iters):
            start = i * offset
            end = start + offset
            if end > query_ranges.shape[0]:
                end = query_ranges.shape[0]
            small_query_ranges = query_ranges[start:end]
            _, small_query_key_dists_list, _ = self.get_query_keys_within_sphericals(small_query_ranges,
                                                                                     points[start:end],
                                                                                     radiuses[start:end])
            for j in range(end - start):
                dists = small_query_key_dists_list[j]
                if dists.shape[0] >= K:
                    radius_list[j + start] = dists[0:K]
            print '**************', i, 'finished*******************'

        return radius_list

    def get_estimate_radiuses(self, lattice_points, tau, K):
        # tau = 100
        radius = (self.max_value_each_dim - self.min_value_each_dim) / tau * 2
        radius_list = self.get_radius_for_knn_query(lattice_points, radius, K)
        print '-----lattice_points.shape =', lattice_points.shape

        while True:
            indices_list = []
            next_lattice_points = []
            for i in range(len(radius_list)):
                dists = radius_list[i]
                if dists is None:
                    indices_list.append(i)
                    next_lattice_points.append(lattice_points[i])

            if len(indices_list) == 0:
                break

            n = len(indices_list)
            next_lattice_points = np.array(next_lattice_points, dtype=np_data_type())
            print 'next_lattice_points.shape =', next_lattice_points.shape
            radius *= 2
            new_radius_list = self.get_radius_for_knn_query(next_lattice_points, radius, K)
            for i in range(len(new_radius_list)):
                dists = new_radius_list[i]
                if dists is not None:
                    n -= 1
                    radius_list[indices_list[i]] = dists
            if n == 0:
                break

        radiuses = np.array(radius_list)
        return radiuses

    def lattice_nodes_gen(self, tau):
        offset = (self.max_value_each_dim - self.min_value_each_dim) / tau
        node_values = []
        for i in range(tau):
            node_values.append(self.min_value_each_dim + i * offset)
        node_values.append(self.max_value_each_dim)
        node_values_list = [np.array(node_values, dtype=np_data_type())] * (self.data_dim)

        lattice_points = np_utils.cartesian_product(node_values_list)
        return lattice_points

    def sampling(self, N):
        return np.random.uniform(low=self.min_value_each_dim, high=self.max_value_each_dim, size=[N, self.data_dim])

    def knn_query(self, points, K, ideal=None):

        def inner_qr_gen(query_points, xradiuses, data_dim):
            xradiuses = np.reshape(xradiuses, [xradiuses.shape[0], 1])
            query_ranges = np.zeros(shape=[query_points.shape[0], data_dim * 2], dtype=np_data_type())
            query_ranges[:, 0:data_dim] = query_points - xradiuses
            query_ranges[:, data_dim:] = query_points + xradiuses
            query_ranges = np.clip(query_ranges, a_min=Config().min_value, a_max=Config().max_value)
            return query_ranges

        start = time.time()
        dists, node_indices_list = self.lat_reg.fit(points)
        end = time.time()
        print 'approximating distance bounds takes', end - start, 'seconds'
        if ideal is not None:
            dists = ideal
            print 'dists.shape =', dists.shape
        dists = np.clip(dists, a_min=1, a_max=Config().max_value)
        radiuses = dists[:, K - 1]

        init_radiuses = np.zeros_like(radiuses, dtype=radiuses.dtype)
        init_radiuses[0:init_radiuses.shape[0]] = radiuses
        # query_ranges = np.zeros(shape=[points.shape[0], self.data_dim * 2], dtype=np_data_type())
        # query_ranges[:, 0:self.data_dim] = points - radiuses
        # query_ranges[:, self.data_dim:] = points + radiuses
        # query_ranges = np.clip(query_ranges, a_min=Config().min_value, a_max=Config().max_value)

        next_indices = np.arange(0, points.shape[0], dtype=np_idx_type())
        next_points = points
        next_radiuses = radiuses

        total_n_pages = 0
        n_pages_every_query = [0] * points.shape[0]

        all_queried_keys = [None] * points.shape[0]
        iter = 0
        while True:
            next_indices_list = []
            next_points_list = []
            next_radiuses_list = []
            # print '-----------------------------iter', iter, '----------------------------'
            query_ranges = inner_qr_gen(next_points, next_radiuses, self.data_dim)
            start = time.time()
            query_keys_list, _, n_pages_list = self.get_query_keys_within_sphericals(query_ranges, next_points,
                                                                                     next_radiuses)
            end = time.time()
            print 'range query takes', end - start, 'seconds'

            for i in range(len(query_keys_list)):
                queied_keys = query_keys_list[i]
                n_queried_keys = queied_keys.shape[0]
                act_i = next_indices[i]

                if n_queried_keys >= K:
                    n_pages_every_query[act_i] = n_pages_list[i]
                    total_n_pages += n_pages_list[i]
                    # all_queried_keys[act_i] = queied_keys[0:K]
                    all_queried_keys[act_i] = queied_keys
                else:
                    next_indices_list.append(act_i)
                    next_points_list.append(next_points[i])

                    radius = next_radiuses[i] * 1.1
                    if n_queried_keys > 0:
                        # radius = next_radiuses[i] * (math.pow(1. * K / n_queried_keys, 1. / self.data_dim) + 0.05)
                        tmp = math.pow(1. * K / n_queried_keys, 1. / self.data_dim)
                        ratio_bd = 1.1
                        if tmp > ratio_bd:
                            tmp = ratio_bd
                        radius = next_radiuses[i] * tmp
                    next_radiuses_list.append(radius)

                    radiuses[act_i] = radius

            if len(next_indices_list) > 0:
                print len(next_indices_list)
                next_indices = np.array(next_indices_list, dtype=np_idx_type())
                next_points = np.array(next_points_list, dtype=np_data_type())
                next_radiuses = np.array(next_radiuses_list, dtype=np_data_type())
            else:
                break

            iter += 1

        return all_queried_keys, total_n_pages, radiuses, init_radiuses, node_indices_list, n_pages_every_query

    def overlap_spherical(self, data, center, radius):
        offsets = data - center
        dists = np.linalg.norm(offsets, axis=1)
        indices = (dists <= radius)
        query_keys = data[indices]
        dists = dists[indices]
        sorted_indices = np.argsort(dists)
        return query_keys[sorted_indices], dists[sorted_indices]

    def overlap(self, data, query_range):
        query_keys = data[(data[:, 0] >= query_range[0]) & (data[:, 0] <= query_range[self.data_dim])]
        for i in range(1, self.data_dim):
            if query_keys.shape[0] > 0:
                query_keys = query_keys[
                    (query_keys[:, i] >= query_range[i]) & (query_keys[:, i] <= query_range[i + self.data_dim])]
            else:
                break
        return query_keys

    def append_data_in_page(self, page, point):
        return np.r_[page, point.reshape([-1, self.data_dim])]

    def split_page(self, page, point, print_flag=False):
        page = self.append_data_in_page(page, point)
        page_mappings = self.monotone_mappings(page)
        idxes = np.argsort(page_mappings)
        page_mappings = page_mappings[idxes]
        page = page[idxes]
        N = int((page.shape[0] + 1) / 2)
        return page[0:N], page[N:], page_mappings[0], page_mappings[N], page_mappings[-1]

    def insert_within_shard(self, point, point_mapping, shard_id):
        shard = self.shard_infos[shard_id]
        shard_page_nos = shard[0]
        shard_split_mappings = shard[1]
        if len(shard_page_nos) == 0:
            new_page = np.reshape(point, [1, -1])
            shard_page_nos.append(len(self.pages))
            self.pages.append(new_page)
            self.m_counts.append(1)
            return

        if len(shard_split_mappings) == 0:
            page_idx_in_this_shard = 0
        else:
            page_idx_in_this_shard = np.searchsorted(shard_split_mappings, point_mapping, side='right')

        page_no = shard_page_nos[page_idx_in_this_shard]
        page = self.pages[page_no]

        if page.shape[0] < self.page_size:
            self.pages[page_no] = self.append_data_in_page(page, point)
            self.m_counts[page_no] += 1
        else:
            page_1, page_2, left_split_mapping, middle_split_mapping, right_split_mapping = self.split_page(page, point)
            self.pages[page_no] = page_1
            self.m_counts[page_no] = page_1.shape[0]
            self.pages.append(page_2)
            self.m_counts.append(page_2.shape[0])

            if page_idx_in_this_shard == len(shard_split_mappings):  # last page
                if page_idx_in_this_shard != 0:
                    shard_split_mappings[page_idx_in_this_shard - 1] = left_split_mapping
                shard_split_mappings.insert(page_idx_in_this_shard, middle_split_mapping)
            elif page_idx_in_this_shard == 0:
                shard_split_mappings[0] = right_split_mapping
                shard_split_mappings.insert(0, middle_split_mapping)
            else:
                shard_split_mappings[page_idx_in_this_shard - 1] = left_split_mapping
                shard_split_mappings[page_idx_in_this_shard] = right_split_mapping
                shard_split_mappings.insert(page_idx_in_this_shard, middle_split_mapping)

            shard_page_nos.insert(page_idx_in_this_shard + 1, len(self.pages) - 1)

    def insert(self, points):
        point_mappings = self.monotone_mappings(points)
        print '---point_mappings.shape =', point_mappings.shape
        offset = 10000
        n = int(point_mappings.shape[0]/offset)
        if n * offset < point_mappings.shape[0]:
            n += 1

        shard_ids = np.zeros(shape=[point_mappings.shape[0]],dtype=np_idx_type())
        for i in range(n):
            start = i * offset
            end = start + offset
            if end > point_mappings.shape[0]:
                end = point_mappings.shape[0]
            shard_ids[start:end] = self.predict_shard_ids(point_mappings[start:end])

        # shard_ids = self.predict_shard_ids(point_mappings)

        print '--shard_ids.shape =', shard_ids.shape
        for i in range(points.shape[0]):
            point = points[i]
            point_mapping = point_mappings[i]
            shard_id = shard_ids[i]
            self.insert_within_shard(point, point_mapping, shard_id)
            if i % 10000 == 0:
                print i, 'finished.'

    def insert_test(self, points):
        point_mappings = self.monotone_mappings(points)
        shard_ids = self.predict_shard_ids(point_mappings)
        page_mappings_list = []
        for page in self.pages:
            page_mappings_list.append(self.monotone_mappings(page))

        err_count = 0
        for i in range(points.shape[0]):
            point_mapping = point_mappings[i]
            shard_id = shard_ids[i]
            page_mappings = page_mappings_list[shard_id]
            left_page_mappings = None
            right_page_mappings = None
            # if shard_id != 0:
            #     left_page_mappings = page_mappings_list[shard_id-1]
            if shard_id != len(page_mappings_list) - 1:
                right_page_mappings = page_mappings_list[shard_id + 1]

            lower_bound = 0
            upper_bound = None
            flag = True
            if shard_id == 0:
                upper_bound = right_page_mappings[0]
                if point_mapping >= upper_bound:
                    flag = False
            elif shard_id == len(page_mappings_list) - 1:
                lower_bound = page_mappings[0]
                if point_mapping < lower_bound:
                    flag = False
            else:
                lower_bound = page_mappings[0]
                upper_bound = right_page_mappings[0]
                if point_mapping >= upper_bound or point_mapping < lower_bound:
                    flag = False

            if flag == False:
                err_count += 1
                print 'i =', i, ', point =', points[
                    i], ', point_mapping =', point_mapping, ', lower_bound =', lower_bound, ', upper_bound =', upper_bound

        print 'err_count =', err_count


    def delete_record_from_page(self, page_no, point):
        page = self.pages[page_no]
        idx = -1
        for i in range(page.shape[0]):
            # diff = np.abs((point - page[i]))
            # if diff.sum() == 0:
            flag = True
            for j in range(self.data_dim):
                if point[j] != page[i][j]:
                    flag = False
                    break
            if flag == True:
                idx = i
                break
        if idx < 0:
            return -1
        if page.shape[0] == 1:
            self.pages[page_no] = np.zeros(shape=[0,self.data_dim],dtype=np_data_type())
            self.m_counts[page_no] = 0
            return 0

        if idx == 0:
            new_page = page[1:]
        elif idx == page.shape[0]-1:
            new_page = page[0:-1]
        else:
            new_page = np.concatenate([page[0:idx], page[idx+1:]], axis=0)
        self.pages[page_no] = new_page
        self.m_counts[page_no] = new_page.shape[0]
        return new_page.shape[0]


    def delete_within_shard(self, point, point_mapping, shard_id):
        shard = self.shard_infos[shard_id]
        shard_page_nos = shard[0]
        shard_split_mappings = shard[1]
        # print '*****', point.shape
        # print len(shard_page_nos)
        # print len(shard_split_mappings)
        if len(shard_page_nos) == 0:
            return

        page_idx_left = np.searchsorted(shard_split_mappings, point_mapping, side='left')
        page_idx_right = np.searchsorted(shard_split_mappings, point_mapping, side='right')


        page_idx = -1
        n_records_left = -1
        for i in range(page_idx_left, page_idx_right + 1):
            page_no = shard_page_nos[i]
            n_records_left = self.delete_record_from_page(page_no, point)
            if n_records_left >= 0:
                page_idx = i
                break

        if n_records_left < 0:
            return

        if len(shard_page_nos) == 1:
            if self.m_counts[shard_page_nos[0]] == 0:
                shard[0] = []
            return

        if len(shard_page_nos) == 2:
            page_no_1 = shard_page_nos[0]
            page_no_2 = shard_page_nos[1]
            n1 = self.m_counts[page_no_1]
            n2 = self.m_counts[page_no_2]
            if (n1 + n2) <= self.page_size:
                new_page = np.concatenate([self.pages[page_no_1], self.pages[page_no_2]], axis=0)
                self.pages[page_no_1] = new_page
                self.pages[page_no_2] = np.zeros(shape=[0, self.data_dim], dtype=np_data_type())
                self.m_counts[page_no_1] = n1 + n2
                self.m_counts[page_no_2] = 0
                shard[0] = shard_page_nos[0:-1]
                shard[1] = []
        else:
            left_idx = -1
            if page_idx == 0:
                if self.m_counts[shard_page_nos[0]] + self.m_counts[shard_page_nos[1]] <= self.page_size:
                    left_idx = 0
                # try:
                #
                # except IndexError:
                #     print '------point =', point, len(shard_page_nos), len(self.m_counts), shard_page_nos
            elif page_idx == len(shard_page_nos) - 1:
                max_idx = len(shard_page_nos) - 1
                if self.m_counts[shard_page_nos[max_idx - 1]] + self.m_counts[shard_page_nos[max_idx]] <= self.page_size:
                    left_idx = max_idx - 1
            else:
                if (self.m_counts[shard_page_nos[page_idx]] + self.m_counts[shard_page_nos[page_idx - 1]] <= self.page_size):
                    left_idx = page_idx - 1
                elif self.m_counts[shard_page_nos[page_idx]] + self.m_counts[shard_page_nos[page_idx + 1]] <= self.page_size:
                    left_idx = page_idx
            if left_idx >= 0:
                page_no_1 = shard_page_nos[left_idx]
                page_no_2 = shard_page_nos[left_idx + 1]
                n1 = self.m_counts[page_no_1]
                n2 = self.m_counts[page_no_2]


                new_page = np.concatenate([self.pages[page_no_1], self.pages[page_no_2]], axis=0)
                self.pages[page_no_1] = new_page
                self.pages[page_no_2] = np.zeros(shape=[0, self.data_dim], dtype=np_data_type())

                self.m_counts[page_no_1] = n1 + n2
                self.m_counts[page_no_2] = 0
                shard_page_nos.remove(page_no_2)
                shard_split_mappings.remove(shard_split_mappings[left_idx])

                shard_page_nos[left_idx] = page_no_1



    def delete(self, points):
        # self.tree.reverse()
        point_mappings = self.monotone_mappings(points)

        offset = 10000
        n = int(point_mappings.shape[0] / offset)
        if n * offset < point_mappings.shape[0]:
            n += 1

        shard_ids = np.zeros(shape=[point_mappings.shape[0]], dtype=np_idx_type())
        for i in range(n):
            start = i * offset
            end = start + offset
            if end > point_mappings.shape[0]:
                end = point_mappings.shape[0]
            shard_ids[start:end] = self.predict_shard_ids(point_mappings[start:end])
        print '--shard_ids.shape =', shard_ids.shape

        for i in range(points.shape[0]):
            # print 'i =', i
            point = points[i]
            point_mapping = point_mappings[i]
            shard_id = shard_ids[i]
            self.delete_within_shard(point, point_mapping, shard_id)
            if i % 10000 == 0:
                print i, 'finished.'


    def save(self):
        FileViewer.detect_and_create_dir(self.model_dir)
        meta_infos_path = os.path.join(self.model_dir, 'meta_infos.npy')
        col_params_path = os.path.join(self.model_dir, 'col_params.npy')
        col_min_mappings_path = os.path.join(self.model_dir, 'col_min_mappings.npy')
        shard_params_path = os.path.join(self.model_dir, 'shard_params.npy')
        page_data_path = os.path.join(self.model_dir, 'page_data.npy')
        m_counts_path = os.path.join(self.model_dir, 'm_counts.npy')
        local_models_path = os.path.join(self.model_dir, 'local_models.pkl')

        np.save(page_data_path, np.concatenate(self.pages, axis=0))
        np.save(m_counts_path, np.array(self.m_counts, dtype=np_idx_type()))

        print 'n_pages =', len(self.m_counts)

        meta_infos = [self.page_size, self.sigma]
        meta_infos.extend(self.col_split_shard_ids.tolist())
        np.save(meta_infos_path, np.array(meta_infos, dtype=np_idx_type()))

        np.save(col_params_path, self.params)
        np.save(col_min_mappings_path, self.col_min_mappings)

        shard_params = np.concatenate([self.Alphas, self.Betas], axis=0)
        print 'shard_params.shape =', shard_params.shape
        np.save(shard_params_path, shard_params)
        with open(local_models_path, 'wb') as writer:
            cPickle.dump(self.shard_infos, writer)

        shard_ids_path = os.path.join(self.model_dir, 'shard_ids.npy')
        np.save(shard_ids_path, self.shard_ids_for_sorted_data)

        col_ids_path = os.path.join(self.model_dir, 'col_ids.npy')
        np.save(col_ids_path, self.col_ids_for_sorted_data)

        all_path = os.path.join(self.model_dir, 'all.npy')
        all_without_addrs_path = os.path.join(self.model_dir, 'all_without_addrs.npy')
        all_params = []
        all_params_without_addrs = []
        empty_flag = -(len(self.m_counts) + 1)

        a = 0
        for shard_id, shard in enumerate(self.shard_infos):
            shard_page_nos = shard[0]
            shard_split_mappings = shard[1]
            if len(shard_page_nos) == 0:
                # continue
                a += 1
                all_params.append(empty_flag)
            else:
                for i in range(len(shard_split_mappings)):
                    all_params.append(shard_page_nos[i])
                    all_params.append(shard_split_mappings[i])
                    all_params_without_addrs.append(shard_split_mappings[i])
                    # a += 1
                all_params.append(shard_page_nos[-1])
                # a +=1
        print '-------a =', a,', page_size =', len(self.m_counts)
        tmp = np.reshape(shard_params, [-1]).tolist()
        all_params.extend(tmp)
        all_params_without_addrs.extend(tmp)
        tmp = self.params.tolist()
        all_params.extend(tmp)
        all_params_without_addrs.extend(tmp)

        all_params = np.array(all_params)
        all_params_without_addrs = np.array(all_params_without_addrs)
        np.save(all_path, all_params)
        np.save(all_without_addrs_path, all_params_without_addrs)

    def load_knn_model(self, lattice_regression_dir):
        self.lat_reg = LatticeRegression()
        self.lat_reg.load_for_fit_only(lattice_regression_dir)

    def check_and_load_params(self):
        all_files = []
        meta_infos_path = os.path.join(self.model_dir, 'meta_infos.npy')
        all_files.append(meta_infos_path)
        params_path = os.path.join(self.model_dir, 'col_params.npy')
        all_files.append(params_path)
        col_min_mappings_path = os.path.join(self.model_dir, 'col_min_mappings.npy')
        all_files.append(col_min_mappings_path)
        shard_params_path = os.path.join(self.model_dir, 'shard_params.npy')
        all_files.append(shard_params_path)
        page_data_path = os.path.join(self.model_dir, 'page_data.npy')
        all_files.append(page_data_path)
        m_counts_path = os.path.join(self.model_dir, 'm_counts.npy')
        all_files.append(m_counts_path)
        local_models_path = os.path.join(self.model_dir, 'local_models.pkl')
        all_files.append(local_models_path)

        for file in all_files:
            if os.path.exists(file) == False:
                return False
        m_counts = np.load(m_counts_path)
        page_data = np.load(page_data_path)

        data_size = page_data.shape[0]
        self.n_pages = m_counts.shape[0]
        self.pages = []
        start = 0

        for i in range(self.n_pages):
            end = start + m_counts[i]
            self.pages.append(page_data[start:end])
            start = end

        self.m_counts = m_counts.tolist()
        print 'n_pages =', len(self.m_counts)

        self.params = np.load(params_path)
        self.params_dump()

        self.col_min_mappings = np.load(col_min_mappings_path)

        meta_infos = np.load(meta_infos_path)
        self.page_size = meta_infos[0]
        self.sigma = meta_infos[1]
        self.col_split_shard_ids = meta_infos[2:]
        self.cal_shard_numbers_each_col()

        shard_params = np.load(shard_params_path)
        n_cols = shard_params.shape[0] / 2
        self.Alphas = shard_params[0:n_cols]
        self.Betas = shard_params[n_cols:]

        with open(local_models_path, 'rb') as reader:
            self.shard_infos = cPickle.load(reader)

        print 'n_shards =', len(self.shard_infos)

        shard_ids_path = os.path.join(self.model_dir, 'shard_ids.npy')
        self.shard_ids_for_sorted_data = np.load(shard_ids_path)

        col_ids_path = os.path.join(self.model_dir, 'col_ids.npy')
        self.col_ids_for_sorted_data = np.load(col_ids_path)

        return True
        # return data_size, page_data

    def cal_page_split_mappings(self):
        n_pages = len(self.pages)
        self.page_split_mappings = np.zeros(shape=[n_pages], dtype=np_data_type())
        for i in range(n_pages - 1):
            page_mappings = self.monotone_mappings(self.pages[i + 1])
            self.page_split_mappings[i] = page_mappings[0]

    def rebuild_sorted_data(self):
        # sorted_data = np.zeros(shape=[data_size, self.data_dim],dtype=np_data_type())
        sorted_data = []
        for shard_id, shard in enumerate(self.shard_infos):
            shard_page_nos = shard[0]
            for page_no in shard_page_nos:
                page = self.pages[page_no]

                page_mappings = self.monotone_mappings(page)
                idxes = np.argsort(page_mappings)
                page = page[idxes]

                sorted_data.append(page)
        sorted_data = np.concatenate(sorted_data, axis=0)
        return sorted_data


def check_order(mappings):
    count = 0
    for i in range(mappings.shape[0] - 1):
        if mappings[i] > mappings[i + 1]:
            print i, mappings[i], mappings[i + 1]
            count += 1
    print '**********count =', count
