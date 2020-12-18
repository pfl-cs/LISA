import numpy as np
import math


def get_split_points_and_idxes(x_data, N, max_value=None, type='data_partition'):
    """
    :param x_data: sorted 1-D data
    :param N: number of parts
    :return:
    """

    x_split_points = []
    if max_value is None:
        max_value = x_data[-1] + 0.01
    n_every_part = x_data.shape[0] / N
    n_remainder = x_data.shape[0] % N

    for i in range(n_remainder):
        if i == N - 1:
            x_split_points.append(max_value)
            continue
        idx = (i + 1) * (n_every_part + 1)
        split_point = x_data[idx - 1]
        x_split_points.append(split_point)

    for i in range(n_remainder, N):
        if i == N - 1:
            x_split_points.append(max_value)
            continue
        idx = (i + 1) * n_every_part + n_remainder
        split_point = x_data[idx - 1]
        x_split_points.append(split_point)

    x_split_idxes = np.searchsorted(x_data, x_split_points, side='left')
    return np.array(x_split_points, dtype=x_data.dtype), x_split_idxes



def partition(data, dim, start, end, n_parts, split_points_list, split_idxes_list, max_value_each_dim):
    part_data = data[start:end]
    one_dim_data = part_data[:, dim]
    sorted_idxes = np.argsort(one_dim_data)
    data[start:end] = part_data[sorted_idxes]
    one_dim_data = data[start:end, dim]

    split_points, split_idxes = get_split_points_and_idxes(one_dim_data, n_parts, max_value=max_value_each_dim)
    split_points_list[dim].append(split_points)
    split_idxes += start
    split_idxes_list[dim].append(split_idxes)

    next_dim_start = start
    for i in range(split_idxes.shape[0]):
        next_dim_end = split_idxes[i]
        if dim < data.shape[1] - 2:
            partition(data, dim + 1, next_dim_start, next_dim_end, n_parts, split_points_list, split_idxes_list, max_value_each_dim)
        next_dim_start = next_dim_end


def create_borders(split_points_list):
    n_parts = split_points_list[0][0].shape[0]
    n_cells = len(split_points_list[-1]) * n_parts

    borders = np.zeros(shape=[n_cells, len(split_points_list)], dtype=split_points_list[0][0].dtype)
    all_cell_measures = np.ones(shape=[n_cells], dtype=borders.dtype)
    dim = 0
    for one_dim_split_points_list in split_points_list:
        n_repeat = (n_cells / n_parts) / len(one_dim_split_points_list)

        start = 0
        for split_points in one_dim_split_points_list:
            front_split_points = np.zeros_like(split_points, dtype=split_points.dtype)
            front_split_points[1:] = split_points[0:-1]
            lens = split_points - front_split_points
            one_dim_front_split_points_list = []
            one_dim_lens_list = []
            for _ in range(n_repeat):
                one_dim_front_split_points_list.append(front_split_points)
                one_dim_lens_list.append(lens)

            tmp = np.reshape(np.array(one_dim_front_split_points_list).transpose(), [-1])
            borders[start:start + tmp.shape[0], dim] = tmp

            tmp = np.reshape(np.array(one_dim_lens_list).transpose(), [-1])

            all_cell_measures[start:start + tmp.shape[0]] *= tmp
            start += tmp.shape[0]
        dim += 1

    return borders, all_cell_measures



def generate_grid_cells(data, n_parts_each_dim, n_models, min_value_each_dim, max_value_each_dim, eta):
    size = data.shape[0]
    n_dim = data.shape[1]

    split_upper_bounds_list = []
    split_idxes_list = []
    for i in range(n_dim - 1):
        split_idxes_list.append([])
        split_upper_bounds_list.append([])

    partition(data, 0, 0, size, n_parts_each_dim, split_upper_bounds_list, split_idxes_list, max_value_each_dim)

    borders, all_cell_measures = create_borders(split_upper_bounds_list)
    print 'borders.shape =', borders.shape

    unsorted_one_dim_data = data[:, -1]
    sorted_idxes = np.argsort(unsorted_one_dim_data)
    sorted_one_dim_data = unsorted_one_dim_data[sorted_idxes]
    last_dim_split_upper_bounds, _ = get_split_points_and_idxes(sorted_one_dim_data, n_parts_each_dim,
                                                                max_value=max_value_each_dim)
    last_dim_front_split_points = np.zeros_like(last_dim_split_upper_bounds, dtype=last_dim_split_upper_bounds.dtype)
    last_dim_front_split_points[1:] = last_dim_split_upper_bounds[0:-1]

    all_cell_ids = np.zeros(shape=[data.shape[0]], dtype=np.int64)
    mappings = np.zeros(shape=[data.shape[0]], dtype=data.dtype)

    second_last_split_idxes_list = split_idxes_list[-1]
    cell_id = 0
    start = 0
    max_measure = 0
    for split_idxes in second_last_split_idxes_list:
        # print 'split_idxes.shape =', split_idxes.shape
        for i in range(split_idxes.shape[0]):
            end = split_idxes[i]
            # print 'start =', start, 'end =', end
            if end > start:
                part_borders = borders[cell_id]
                part_cell_measures = all_cell_measures[cell_id]
                part_data = data[start:end]
                part_measures = np.prod(part_data[:, 0:-1] - part_borders, axis=1) / part_cell_measures
                # print 'haha', part_data.shape, part_measures.shape
                tmp = part_measures.max()
                if tmp > max_measure:
                    max_measure = tmp
                all_cell_ids[start:end] = cell_id
                part_mappings = (part_measures * eta) + (part_data[:,-1] / max_value_each_dim * (n_parts_each_dim - 1)) + (cell_id * n_parts_each_dim)
                part_idxes = np.argsort(part_mappings)
                data[start:end] = part_data[part_idxes]
                mappings[start:end] = part_mappings[part_idxes]

                start = end
            cell_id += 1

    max_mapping = math.pow(n_parts_each_dim, data.shape[1])
    split_mappings = np.zeros(shape=[n_models], dtype=data.dtype)
    split_mappings[-1] = max_mapping + 1

    offset = int(mappings.shape[0] / n_models)
    for i in range(1, n_models):
        idx = i * offset
        m = (mappings[idx] + mappings[idx - 1]) / 2
        split_mappings[i - 1] = m

    split_idxes = np.searchsorted(mappings, split_mappings, side='right')

    model_split_mappings = np.zeros(shape=[split_idxes.shape[0]])
    model_split_mappings[0:-1] = mappings[split_idxes[0:-1]]
    model_split_mappings[-1] = max_mapping + 1

    params = []
    for one_dim_split_upper_bounds_list in split_upper_bounds_list:
        for split_upper_bounds in one_dim_split_upper_bounds_list:
            params.extend(split_upper_bounds.tolist())

    params.extend(last_dim_split_upper_bounds.tolist())

    params.extend(model_split_mappings.tolist())
    params.append(eta)
    params.append(min_value_each_dim)
    params.append(max_value_each_dim)
    params.append(n_models)
    params.append(n_parts_each_dim + 0.5)
    params.append(n_dim + 0.5)

    check_order(mappings)

    return data, mappings, np.array(params,dtype=np.float64), all_cell_measures



def check_order(mappings):
    count = 0
    for i in range(mappings.shape[0] - 1):
        if mappings[i] > mappings[i + 1]:
            print i, mappings[i], mappings[i + 1]
            count += 1
    print '**********count =', count

