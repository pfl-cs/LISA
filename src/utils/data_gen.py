import sys

sys.path.append("../../")
from core.config import Config
import FileViewer
import os
import random
from src.FLAGS_DEFINE import *
import cv2



def remove_repeated(data):
    print data.dtype
    data_list = data.tolist()
    print data_list[0:5]
    print len(data_list)

    data_str_list = [','.join([str(j) for j in i]) for i in data_list]
    print len(data_str_list)
    print data_str_list[0:5]
    data_str_set = set(data_str_list)
    print len(data_str_set)
    data_list = [[float(j) for j in i.split(',')] for i in data_str_set]
    print data_list[0:5]
    print len(data_list)
    data = np.array(data_list)
    print data.dtype
    return data


def synthetic_data_gen(low, high, n, data_dim=2, max_value=10000, dist_type='uniform', zipf_a=1.001):
    if dist_type == 'zipf':
        all_data_set = set()
        iter = 0
        while True:
            l = []
            n_min = -1
            for i in range(data_dim):
                x = np.random.zipf(a=zipf_a, size=[1000000, data_dim])
                x = x[x < max_value]
                if n_min < 0 or n_min > x.shape[0]:
                    n_min = x.shape[0]
                l.append(x)
            for i in range(len(l)):
                l[i] = l[i][0:n_min]
            x = np.array(l).transpose()
            mapping = x[:,0]
            for i in range(1,data_dim):
                mapping = mapping * 10000 + x[:,i]

            data_list = mapping.tolist()
            all_data_set = all_data_set.union(data_list)
            iter += 1
            # if iter > 2:
            #     break
            print 'iter =', iter, 'size =', len(all_data_set)
            if len(all_data_set) > n:
                break



        data = np.zeros(shape=[len(all_data_set), data_dim], dtype=np.int64)
        mapping = np.array(list(all_data_set), dtype=np.int64)
        for i in range(data_dim):
            data[:,data_dim-1-i] = mapping % 10000
            mapping /= 10000

        return data.astype(np_data_type())

    else:
        data = np.random.uniform(low=low, high=high, size=[n, data_dim])
        # data = np.random.random_sample(size=[n, 2]) * (high - low) + low

        return data.astype(np_data_type())

def flatten(data, min_value=0.01, max_value=10000):
    print data.dtype
    data_dim = data.shape[1]
    min_values = data.min(axis=0)
    max_values = data.max(axis=0)

    range_min_values = np.array([min_value] * data_dim, dtype=np_data_type())
    range_max_values = np.array([max_value] * data_dim, dtype=np_data_type())
    ranges = range_max_values - range_min_values
    data = (data - min_values) * (ranges / (max_values - min_values)) + range_min_values
    print data.min(axis=0)
    print data.max(axis=0)
    return data




def get_feature_for_single_img(img):
    res = []
    h_start = 25
    w_start = 25

    h_end = 450
    w_end = 600
    # 450 * 600
    for h in range(h_start, h_end, 50):
        for w in range(w_start, w_end, 50):
            res.append(img[h, w, 0])
            res.append(img[h, w, 1])
            res.append(img[h, w, 2])
    return res

def imagenet_data_gen(raw_imagenet_dir, data_dir):

    raw_data_dir = os.path.join(data_dir, 'raw')
    raw_unresized_data_path = os.path.join(raw_data_dir, 'raw_unresized.npy')
    raw_resized_path = os.path.join(raw_data_dir, 'raw_resized.npy')
    raw_data_int_path = os.path.join(raw_data_dir, 'raw_data_int.npy')
    data_1_path = os.path.join(data_dir, 'data_1.npy')
    data_2_path = os.path.join(data_dir, 'data_2.npy')

    file_paths = FileViewer.list_files(raw_imagenet_dir, 'JPEG')
    print 'len(file_paths) =', len(file_paths)
    step = 10000
    N = len(file_paths) / step

    curr_i = 9
    # assert N >= curr_i
    # for i in range(N + 1):
    for i in range(curr_i, curr_i + 1):
    # for i in range(N + 1):
        start = i * step
        end = start + step
        if end > len(file_paths):
            end = len(file_paths)

        data_path = os.path.join(raw_imagenet_dir, 'data_' + str(i) + '.npy')
        features = []
        for j in range(start, end):
            path = file_paths[j]
            img = cv2.imread(path)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.resize(img, (600, 450))
                features.append(get_feature_for_single_img(img))

        features = np.array(features, dtype=np.int64)
        np.save(data_path, features)

    raw_unresized_data = []
    npy_paths = FileViewer.list_files(raw_imagenet_dir, 'npy')
    for path in npy_paths:
        part_data = np.load(path)
        raw_unresized_data.append(part_data)
    #
    raw_unresized_data = np.concatenate(raw_unresized_data, axis=0)
    np.save(raw_unresized_data_path, raw_unresized_data)
    raw_unresized_data = np.load(raw_unresized_data_path)
    print 'raw_unresized_data.shape =', raw_unresized_data.shape
    print 'raw_unresized_data.dtype =', raw_unresized_data.dtype
    raw_resized_data = np.zeros(shape=[raw_unresized_data.shape[0] * raw_unresized_data.shape[1] / 6, 6],dtype=np.int64)

    count = 0
    curr_idx = 0
    step = raw_unresized_data.shape[0]
    n_parts = raw_unresized_data.shape[1] / 6
    for j in range(n_parts):
        raw_resized_data[curr_idx:curr_idx + step, 0:3] = raw_unresized_data[:, j * 3: j * 3 + 3]
        idx = raw_unresized_data.shape[1] - (j + 1) * 3
        raw_resized_data[curr_idx:curr_idx + step, 3:] = raw_unresized_data[:, idx: idx + 3]
        count += 1
        curr_idx += step
    print 'count =', count
    np.save(raw_resized_path, raw_resized_data)

    raw_resized_data = np.load(raw_resized_path)
    print 'raw_resized_data.shape =', raw_resized_data.shape
    print 'raw_resized_data.dtype =', raw_resized_data.dtype

    raw_resized_list = raw_resized_data.tolist()
    print 'len(raw_resized_list) =', len(raw_resized_list)

    data_str_set = set()
    for l in raw_resized_list:
        s = '-'.join([str(i) for i in l])
        data_str_set.add(s)

    print 'data_str_set.size =', len(data_str_set)


    raw_data_int = []
    for s in data_str_set:
        l = [int(i) for i in s.split('-')]
        raw_data_int.append(l)
    raw_data_int = np.array(raw_data_int, dtype=np.int64)
    np.save(raw_data_int_path, raw_data_int)

    raw_data_int = np.load(raw_data_int_path)
    print 'raw_data_int.max, raw_data_int.min =', raw_data_int.max(), raw_data_int.min()
    noise = np.random.uniform(0.0, 1.0, size=raw_data_int.shape)
    print 'noise.max, noise.min =', noise.max(), noise.min()
    print 'noise.shape =', noise.shape
    raw_data = (noise + raw_data_int) / 257.0  * 1000.0 + 0.001
    print raw_data.max(), raw_data.min()
    idxes = np.arange(0, raw_data.shape[0], dtype=np.int64).tolist()
    random.shuffle(idxes)
    raw_data = raw_data[idxes]
    data_0 = raw_data[0:40000000]
    data_2 = raw_data[40000000:]

    random.shuffle(idxes)
    raw_data = raw_data[idxes]
    data_3 = raw_data[0:40000000]

    return data_0, data_2, data_3



if __name__ == '__main__':
    workspace = '/home/pfl/workspace/LISA/4d_uniform'
    Config(workspace)
    data_dir = Config().data_dir
    print '----data_dir =', data_dir
    FileViewer.detect_and_create_dir(data_dir)


    N = 100000000
    raw_data = synthetic_data_gen(Config().min_value+0.5, Config().max_value-0.5, N+1000000, data_dim=Config().data_dim, dist_type='uniform')
    raw_data = np.round(raw_data, 5)
    raw_remove_repated_data = remove_repeated(raw_data)
    data = raw_remove_repated_data[0:N]
    assert(raw_remove_repated_data.shape[0] >= N)
    print 'data.shape =', data.shape

    idxes = np.arange(0, data.shape[0], dtype=np.int64).tolist()
    random.shuffle(idxes)
    data = data[idxes]
    halfN = N / 2
    data_0 = data[0:halfN]
    data_2 = data[halfN:]


    random.shuffle(idxes)
    data = data[idxes]
    data_3 = data[halfN:]
    print 'data_3.shape =', data_3.shape


    static_data_path = os.path.join(data_dir, Config().static_data_name)
    data_to_insert_path = os.path.join(data_dir, Config().data_to_insert_name)
    data_to_delete_path = os.path.join(data_dir, Config().data_to_delete_name)
    np.save(static_data_path, data_0)
    np.save(data_to_insert_path, data_2)
    np.save(data_to_delete_path, data_3)



    #-----------imagenet----------
    # raw_imagenet_dir = '/home/pfl/ImageNet'
    # home_dir = '/home/pfl/workspace/LISA/imagenet/'
    # data_dir = os.path.join(home_dir, 'data')
    # data_0, data_2, data_3 = imagenet_data_gen(raw_imagenet_dir, data_dir)
    # static_data_path = os.path.join(data_dir, Config().static_data_name)
    # data_to_insert_path = os.path.join(data_dir, Config().data_to_insert_name)
    # data_to_delete_path = os.path.join(data_dir, Config().data_to_delete_name)
    # np.save(static_data_path, data_0)
    # np.save(data_to_insert_path, data_2)
    # np.save(data_to_delete_path, data_3)


