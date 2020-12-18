import numpy as np
import scipy
import sys

sys.path.append('../..')
from src.utils.core.config import Config
from src.FLAGS_DEFINE import *
from src.utils import np_utils
from src.utils import FileViewer
import os
import threading

def check_order(mappings):
    count = 0
    for i in range(mappings.shape[0] - 1):
        if mappings[i] >= mappings[i + 1]:
            print i, mappings[i], mappings[i + 1]
            count += 1
    print '**********count =', count

class PiecewiseModel:
    def __init__(self, id, sorted_mappings, sigma=100):
        self.id = id
        self.min_value = sorted_mappings.min()
        # print 'min_Value =', self.min_value
        self.sorted_mappings = sorted_mappings - self.min_value
        # self.sorted_mappings = sorted_mappings
        self.positions = np.arange(0, self.sorted_mappings.shape[0], dtype=np.int64)
        self.sigma = sigma
        self.alphas = None
        self.betas = np.zeros(shape=[self.sigma], dtype=np.float64)

        self.init_alphas = np.zeros(shape=[self.sigma], dtype=np.float64)
        self.init_betas = np.zeros(shape=[self.sigma], dtype=np.float64)
        self.sorted_mappings_reshape = np.reshape(self.sorted_mappings, [-1,1])

    @staticmethod
    def relu(A):
        A[A<0] = 0
        return A

    def cal_alphas(self, betas, mappings=None, positions=None):
        if mappings is None or positions is None:
            mappings = self.sorted_mappings_reshape
            positions = self.positions
        A = self.relu(np.tile(mappings, [1, self.sigma]) - betas.transpose())
        symm = np.matmul(A.transpose(), A)
        # print 'symm.shape =', symm.shape
        if (np.linalg.cond(symm) < 1 / sys.float_info.epsilon):
            left_part = np.linalg.inv(symm)
            right_part = A.transpose().dot(positions)
            alphas = left_part.dot(right_part)
            # alphas = np.linalg.lstsq(A, positions)[0]
            # print alphas
            return alphas, A
        else:
            # print '---------------'
            return None, None

        # alphas = np.linalg.lstsq(A, positions, rcond=None)[0]
        return alphas, A

    def predict_sorted_mappning_idxes(self, A=None, alphas=None):
        if A is None or alphas is None:
            A = self.A
            alphas = self.alphas
        pred_idxes = A.dot(alphas)
        a_max = pred_idxes.max()
        pred_idxes = np.clip(pred_idxes, a_min=0, a_max=a_max)

        return pred_idxes

    def predict_idxes(self, mappings=None, betas=None, alphas=None):
        if betas is None:
            betas = self.betas
        if mappings is None:
            mappings = self.sorted_mappings
        if alphas is None:
            alphas = self.alphas
        A = self.relu(np.tile(np.reshape(mappings, [-1, 1]), [1, self.sigma]) - betas.transpose())
        # A = self.relu(np.tile(self.sorted_mappings_reshape, [1, self.sigma]) - betas.transpose())
        pred_idxes = A.dot(alphas)
        return pred_idxes

    def cal_loss(self, A, alphas):
        # A = self.relu(np.tile(self.sorted_mappings, [1, self.sigma]) - betas.transpose())
        # r = A.dot(alphas).clip(min=0, max=self.sorted_mappings.shape[0] - 1) - self.positions
        # r = A.dot(alphas).clip(min=0, max=self.sorted_mappings.shape[0]) - self.positions
        r = A.dot(alphas).clip(min=0, max=self.sorted_mappings.shape[0]) - self.positions

        # print '-----r.max =', np.abs(r).max()
        return np.sum(r*r)


    def loss(self):
        A = self.relu(np.tile(self.sorted_mappings_reshape, [1, self.sigma]) - self.betas.transpose())
        r = A.dot(self.alphas).clip(min=0,max=self.sorted_mappings.shape[0])-self.positions
        return np.sum(r*r)


    def lr_search(self, s, init_betas, init_loss):
        init_lr = 1.

        # lrs = [0.00001, 0.0001, 0.01, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8]
        lrs = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8]
        # lrs = [0, 001, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8]
        # lrs = []
        # # for i in range(1, 6):
        # #     lrs.append(i * 0.2 * init_lr)
        # # lrs.extend([0.025, 0.01, 0.001, 0.0005, 0.0001, 1e-5, 1e-6, 4, 8, 16])
        # lrs.extend([0,001, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8])

        # for i in range(8, 20):
        #     lrs.append(i * 0.05 * init_lr)
        # lrs.extend([0.01, 8])

        losses = []
        beta_list = []
        alpha_list = []


        for lr in lrs:
            betas = init_betas + lr * s
            betas = np.sort(betas)
            # betas[0] = init_betas[0]

            alphas, A = self.cal_alphas(betas)
            # alphas_2 = self.cal_alphas_with_monotone_constrain(betas, alphas)
            if alphas is None or A is None:
                loss = init_loss + 1
            else:
                alphas_cumsum = np.cumsum(alphas)
                if alphas_cumsum.min() > 0:
                    loss = self.cal_loss(A, alphas)
                else:
                    loss = init_loss + 1
            losses.append(loss)
            beta_list.append(betas)
            alpha_list.append(alphas)

        losses = np.array(losses)
        lrs = np.array(lrs)
        idx = np.argmin(losses,axis=0)
        min_loss = losses[idx]
        if min_loss < init_loss:
            return lrs[idx], min_loss, beta_list[idx], alpha_list[idx]
        else:
            return -1, -1, None, None

    def cal_alphas_with_monotone_constrain(self, betas, old_alphas):
        pred_positions = self.predict_idxes(betas, betas, old_alphas).clip(min=0, max=self.sorted_mappings.shape[0])
        pred_positions = np.sort(pred_positions)
        assert (abs(pred_positions[0]) < 1e-4)

        alphas = np.zeros(shape=[self.sigma], dtype=np.float64)
        for i in range(1, pred_positions.shape[0]):
            v = 0
            beta_i = betas[i]
            for j in range(i - 1):
                v += alphas[j] * (beta_i - betas[j])
            alphas[i - 1] = (pred_positions[i] - v) / (beta_i - betas[i - 1])

        max_mapping = self.sorted_mappings[-1]
        v = 0
        for j in range(1, self.sigma):
            v += alphas[j] * (betas[j] - betas[j - 1])

        alphas[-1] = (self.sorted_mappings.shape[0] - 1) / (max_mapping - betas[-1])
        alphas_cumsum = np.cumsum(alphas)
        if alphas_cumsum[-1] < 0:
            print '**************'
            alphas[-1] = -alphas_cumsum[-2]

        # print 'alphas =', alphas.tolist()
        all_pred_idxes = self.predict_idxes(self.sorted_mappings, betas, alphas).clip(min=0, max=self.sorted_mappings.shape[0])
        act_idxes = np.arange(0, self.sorted_mappings.shape[0], dtype=np_data_type())
        diff = (all_pred_idxes - act_idxes)
        # print 'all_loss =', np.sum(diff * diff)
        print all_pred_idxes[0:100].tolist()

        return alphas


    def cal_init_alphas(self, betas):
        idxes = np.searchsorted(self.sorted_mappings, betas, side= 'right')

        pred_positions = (idxes - 0.5).clip(min=0)
        alphas = np.zeros(shape=[self.sigma], dtype=np.float64)
        for i in range(1, pred_positions.shape[0]):
            v = 0
            beta_i = betas[i]
            for j in range(i - 1):
                v += alphas[j] * (beta_i - betas[j])
            diff = (beta_i - betas[i - 1])
            if diff <= 0:
                alphas[i - 1] = 0
            else:
                alphas[i - 1] = (pred_positions[i] - v) / diff

        max_mapping = self.sorted_mappings[-1]
        v = 0
        for j in range(1, self.sigma):
            v += alphas[j] * (betas[j] - betas[j - 1])

        alphas[-1] = (self.sorted_mappings.shape[0] - 1) / (max_mapping - betas[-1])
        alphas_cumsum = np.cumsum(alphas)
        if alphas_cumsum[-1] < 0:
            print '**************'
            alphas[-1] = -alphas_cumsum[-2]

        return alphas



    def train(self):
        self.betas = np.zeros(shape=[self.sigma], dtype=np.float64)
        self.alphas = np.zeros(shape=[self.sigma], dtype=np.float64)



    def train2(self):
        n_each_cell = int(self.sorted_mappings.shape[0] / self.sigma)
        split_idxes = np.arange(0, self.sigma, dtype=np.int64) * n_each_cell
        self.betas = self.sorted_mappings[split_idxes].reshape([-1])
        self.init_betas = self.sorted_mappings[split_idxes].reshape([-1])
        self.init_alphas = self.cal_init_alphas(self.init_betas)


        # print '----------check betas-----------'
        # check_order(self.betas)
        # print '----------check betas finished-----------'

        k = 0

        while True:
            # print '---------------col_id =', self.id, ', k =', k, '----------------'

            betas = self.betas
            # assert betas is not None
            alphas_1, A = self.cal_alphas(betas)
            # assert A is not None
            if A is None or alphas_1 is None:
                break
            init_loss_1 = self.cal_loss(A, alphas_1)


            init_loss = init_loss_1
            alphas = alphas_1
            if self.check_if_alphas_and_betas_valid(alphas_1, betas) == False:
                # print '****************hahaha'
                alphas = self.cal_alphas_with_monotone_constrain(betas, alphas_1)
                init_loss = self.cal_loss(A, alphas)
                # alphas = alphas_2
                # init_loss = init_loss_2
                # assert self.check_if_alphas_and_betas_valid(alphas, betas) == True
                if self.check_if_alphas_and_betas_valid(alphas, betas) == False:
                    self.alphas = None
                    self.betas = None

            # alphas, A = self.cal_alphas(betas)
            # init_loss = self.cal_loss(A, alphas, if_print=True)
            # else:
            #     alphas = alphas_1
            #     if init_loss_1 > init_loss_2:
            #         alphas = alphas_2
            #         init_loss = init_loss_2
            #         assert self.check_if_alphas_and_betas_valid(alphas, betas) == True

            # alphas = alphas_1
            # if init_loss_1 > init_loss_2 and k >1:
            #     alphas = alphas_2
            #     init_loss = init_loss_2
            #     assert self.check_if_alphas_and_betas_valid(alphas, betas) == True

            G = -np.sign(A).transpose()
            r = A.dot(alphas).clip(min=0,max=self.positions.shape[0]) - self.positions
            K = np.diag(alphas)
            g = 2 * K.dot((G.dot(r))) / self.sorted_mappings.shape[0]
            G_square = np.matmul(G, G.transpose())
            Y = 2 * np.matmul(np.matmul(K, G_square), K) / self.sorted_mappings.shape[0]

            # try:
            #     s = -np.linalg.inv(Y).dot(g)
            # except:
            #     s = -g

            second_grad_flag = True
            if np.linalg.cond(Y) < 1 / sys.float_info.epsilon:
                s = -np.linalg.inv(Y).dot(g)
            else:
                # print '------'
                second_grad_flag = False
                s = -g

            # print 's.shape =', s.shape
            lr, loss, tmp_betas, tmp_alphas = self.lr_search(s, betas, init_loss)

            if lr > 0:
                self.betas = tmp_betas
                self.alphas = tmp_alphas
                # betas += lr * s
                # self.betas = np.sort(betas)
                # self.alphas, _ = self.cal_alphas(self.betas)

            else:
                if second_grad_flag == False:
                    # self.A = A
                    # self.alphas = alphas
                    # print 'loss =', init_loss
                    break
                else:
                    s = -g
                    # lr, loss = self.lr_search(s, betas, init_loss)
                    lr, loss, tmp_betas, tmp_alphas = self.lr_search(s, betas, init_loss)

                    if lr > 0:
                        self.betas = tmp_betas
                        self.alphas = tmp_alphas
                        # betas += lr * s
                        # self.betas = np.sort(betas)
                        # self.alphas, _ = self.cal_alphas(self.betas)
                    else:
                        # print 'loss =', init_loss

                        # all_pred_idxes = self.predict_idxes(self.sorted_mappings)
                        # diff = np.abs(all_pred_idxes - self.positions)
                        # print 'avg_diff =', np.average(diff)
                        break

            if k % 100 == 0:
                print '---------------col_id =', self.id, ', k =', k, '----------------'
                print 'xloss =', loss

            k += 1
            if k >= 400:
                break


    def check_if_valid(self):
        print '*************************'
        epison = 1e-6
        np_utils.check_order(self.predict_idxes(self.betas, self.betas))
        print '*************************'
        y_pred = self.A.dot(self.alphas)
        np_utils.check_order(y_pred)

    @staticmethod
    def check(model_dir):
        alphas_path = os.path.join(model_dir, 'alphas.npy')
        betas_path = os.path.join(model_dir, 'betas.npy')
        alphas = np.load(alphas_path)
        betas = np.load(betas_path)

        # betas_diff = betas[1:] - betas[0:-1]
        # flag = (betas_diff.min() > 0)
        #
        # if flag == False:
        #     print '************************'
        #     return flag
        # alphas_cumsum = np.cumsum(alphas)
        # min_slope = alphas_cumsum.min()
        #
        # return (min_slope >= 0)
        return PiecewiseModel.check_if_alphas_and_betas_valid(alphas, betas)

    @staticmethod
    def check_if_alphas_and_betas_valid(alphas, betas):
        betas_diff = betas[1:] - betas[0:-1]
        flag = (betas_diff.min() > 0)

        if flag == False:
            print '************betas is error'
            return flag
        alphas_cumsum = np.cumsum(alphas)
        min_slope = alphas_cumsum.min()
        # print 'min_slope =', min_slope

        return (min_slope >= 0)


    def save(self, model_dir):
        # meta_data = [self.min_value]
        alphas_path = os.path.join(model_dir, 'alphas.npy')
        betas_path = os.path.join(model_dir, 'betas.npy')
        if self.alphas is not None and self.betas is not None:
            np.save(alphas_path, self.alphas)
            np.save(betas_path, self.betas)
        else:
            np.save(alphas_path, self.init_alphas)
            np.save(betas_path, self.init_betas)

    def load(self, model_dir):
        # meta_data = [self.min_value]
        alphas_path = os.path.join(model_dir, 'alphas.npy')
        betas_path = os.path.join(model_dir, 'betas.npy')
        self.alphas = np.load(alphas_path)
        self.betas = np.load(betas_path)


class myThread(threading.Thread):
    def __init__(self, thread_id, sigma, col_ids, mappings_list, params_dir):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.sigma = sigma
        self.col_ids = col_ids
        self.sorted_mappings_list = mappings_list
        self.params_dir = params_dir

    #
    def run(self):
        for i in range(len(self.col_ids)):
            col_id = self.col_ids[i]
            sorted_mappings = self.sorted_mappings_list[i]
            pm = PiecewiseModel(sorted_mappings,self.sigma)
            model_dir = os.path.join(self.params_dir, str(col_id))
            try:
                pm.train()
                FileViewer.detect_and_create_dir(model_dir)
                pm.save(model_dir)
                print 'thread_id =', self.thread_id, ', model', col_id, 'has been trained'
            except:
                FileViewer.detect_and_delete_dir(model_dir)
                print 'thread_id =', self.thread_id, ', model', col_id, 'encountered exception'



if __name__ == '__main__':
    Config(Config().home_dir)

    one_dim_mappings = np.load(os.path.join(Config().data_dir, 'one_dim_mappings.npy'))
    print '----------dtype =', one_dim_mappings.dtype, one_dim_mappings.shape
    print one_dim_mappings.max(), one_dim_mappings.min()
    col_split_idxes = np.load(os.path.join(Config().data_dir, 'col_split_idxes.npy'))
    print col_split_idxes[0:10]
    print col_split_idxes[-10:]
    print '-------', col_split_idxes.shape
    n_cols = col_split_idxes.shape[0]
    print '----n_cols =', n_cols
    # n_cells_each_cols = one_dim_mappings.shape[0] / n_cols / (Config().page_size * 0.8)

    sigma = Config().sigma
    params_dir = os.path.join(Config().models_dir, 'piecewise')
    params_dir = os.path.join(params_dir, 'cols')
    FileViewer.detect_and_create_dir(params_dir)

    # n_threads = 32
    # mappings_list = [[]] * n_threads
    # col_ids = [[]] * n_threads
    #
    # start = 0
    # for i in range(col_split_idxes.shape[0]):
    #     end = col_split_idxes[i]
    #     one_dim_input = one_dim_mappings[start:end]
    #     mappings_list[i % n_threads].append(one_dim_input)
    #     col_ids[i % n_threads].append(i)
    #     start = end
    #
    # threads = []
    # for i in range(n_threads):
    #     t = myThread(i, sigma, col_ids[i], mappings_list[i], params_dir)
    #     threads.append(t)
    #
    # for t in threads:
    #     t.start()

    # for i in range(646, 647):
    start = 0
    # for i in range(1,1):
    for i in range(849, col_split_idxes.shape[0]):
    # for i in range(151, 151):
    # for i in range(1167, 500, -1):
    # for i in range(col_split_idxes.shape[0] - 1, -1, -1):
    # for i in range(col_split_idxes.shape[0] - 1, 700, -1):
    # for i in range(col_split_idxes.shape[0]):
        # for i in range(col_split_idxes.shape[0], col_split_idxes.shape[0]):
        if i > 0:
            start = col_split_idxes[i - 1]
        end = col_split_idxes[i]
        one_dim_input = one_dim_mappings[start:end]
        pm = PiecewiseModel(i, one_dim_input, sigma)
        model_dir = os.path.join(params_dir, str(i))
        if os.path.exists(model_dir) == False:
            # print '---i =', i
            # try:
            #     pm.train()
            #     FileViewer.detect_and_create_dir(model_dir)
            #     pm.save(model_dir)
            #     print ', model', i, 'has been trained'
            # except:
            #     FileViewer.detect_and_delete_dir(model_dir)
            #     print ', model', i, 'encountered exception'

            pm.train()
            FileViewer.detect_and_create_dir(model_dir)
            pm.save(model_dir)
            print ', model', i, 'has been trained'



    # start = 0
    # # for i in range(1,1):
    # # for i in range(259, col_split_idxes.shape[0]):
    # # for i in range(151, 151):
    # # for i in range(1167, 500, -1):
    # # for i in range(col_split_idxes.shape[0]-1,-1,-1):
    # for i in range(col_split_idxes.shape[0]):
    # # for i in range(col_split_idxes.shape[0], col_split_idxes.shape[0]):
    #     if i > 0:
    #         start = col_split_idxes[i-1]
    #     end = col_split_idxes[i]
    #     one_dim_input = one_dim_mappings[start:end]
    #     pm = PiecewiseModel(i, one_dim_input, sigma)
    #     model_dir = os.path.join(params_dir, str(i))
    #
    #     # pm.train()
    #     # FileViewer.detect_and_create_dir(model_dir)
    #     # pm.save(model_dir)
    #     # print ', model', i, 'has been trained'
    #     try:
    #         pm.train()
    #         FileViewer.detect_and_create_dir(model_dir)
    #         pm.save(model_dir)
    #         print ', model', i, 'has been trained'
    #     except:
    #         FileViewer.detect_and_delete_dir(model_dir)
    #         print ', model', i, 'encountered exception'
    #
    #     # pred_idxes = pm.predict_idxes()
    #     # print pred_idxes[-50:]
    #     # pred_idxes = (pred_idxes / Config().page_size).astype(np.int64)
    #     # max_cell_id = pred_idxes.max()
    #     # entries_count = [0] * (max_cell_id + 1)
    #     # for i in range(pred_idxes.shape[0]):
    #     #     idx = int(pred_idxes[i])
    #     #     entries_count[idx] += 1
    #     # print entries_count
    #     # print max_cell_id
    #
    #     # pm.train()
    #     # FileViewer.detect_and_create_dir(model_dir)
    #     # pm.save(model_dir)
    #
    # start = 0
    # for i in range(col_split_idxes.shape[0]):
    #     end = col_split_idxes[i]
    #     model_dir = os.path.join(params_dir, str(i))
    #     flag = PiecewiseModel.check(model_dir)
    #     if flag == False:
    #         print '--------------------i =', i, '----------------------'
    #         one_dim_input = one_dim_mappings[start:end]
    #         pm = PiecewiseModel(i, one_dim_input, sigma)
    #         # try:
    #         #     # FileViewer.detect_and_delete_dir(model_dir)
    #         #     pm.train()
    #         #     FileViewer.detect_and_create_dir(model_dir)
    #         #     pm.save(model_dir)
    #         #     print ', model', i, 'has been trained'
    #         # except:
    #         #     FileViewer.detect_and_delete_dir(model_dir)
    #         #     print ', model', i, 'encountered exception'
    #
    #         # pm.train()
    #         # FileViewer.detect_and_create_dir(model_dir)
    #         # pm.save(model_dir)
    #
    #     start = end
    #
    # print '---------------'
    #
    # for i in range(col_split_idxes.shape[0]):
    # # for i in range(309, col_split_idxes.shape[0]):
    #     if i == 0:
    #         start = 0
    #     else:
    #         start = col_split_idxes[i-1]
    #     end = col_split_idxes[i]
    #     model_dir = os.path.join(params_dir, str(i))
    #     flag = PiecewiseModel.check(model_dir)
    #     if flag == False:
    #         print '--------------------i =', i, '----------------------'
    #         # one_dim_input = one_dim_mappings[start:end]
    #         # pm = PiecewiseModel(i, one_dim_input, sigma)
    #         # pm.train()
    #         # FileViewer.detect_and_create_dir(model_dir)
    #         # pm.save(model_dir)
    #
    #         # try:
    #         #     # FileViewer.detect_and_delete_dir(model_dir)
    #         #     pm.train()
    #         #     FileViewer.detect_and_create_dir(model_dir)
    #         #     pm.save(model_dir)
    #         #     print ', model', i, 'has been trained'
    #         # except:
    #         #     # FileViewer.detect_and_delete_dir(model_dir)
    #         #     print ', model', i, 'encountered exception'
    #
    #     # one_dim_input = one_dim_mappings[start:end]
    #     # pm = PiecewiseModel(i, one_dim_input, sigma)
    #     # pm.load(model_dir)
    #     # loss = pm.loss()
    #     # # if loss > 3.9e7:
    #     # if loss > 1e8:
    #     #     print '------------------------i =', i, 'loss =', loss
    #     #     try:
    #     #         pm.train()
    #     #         FileViewer.detect_and_create_dir(model_dir)
    #     #         pm.save(model_dir)
    #     #         print 'model', i, 'has been trained'
    #     #     except:
    #     #         # FileViewer.detect_and_delete_dir(model_dir)
    #     #         print 'model', i, 'encountered exception'
    #
    #
    #     start = end
    #
    # print '---------------'
