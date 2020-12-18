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
from scipy.sparse import csr_matrix, csc_matrix, save_npz, load_npz
from scipy.sparse.linalg import inv, spsolve, lsqr
import math
import time



class LatticeRegression:
    def load_training_data(self, lattice_dir):
        lattice_nodes_path = os.path.join(lattice_dir, 'lattice_nodes.npy')
        lattice_training_points_path = os.path.join(lattice_dir, 'training_points.npy')
        lattice_training_radiuses_path = os.path.join(lattice_dir, 'training_radiuses.npy')
        self.A = np.load(lattice_nodes_path).transpose() # m
        self.training_X = np.load(lattice_training_points_path).transpose()
        self.training_Y = np.load(lattice_training_radiuses_path).transpose()

        print 'self.A.shape =', self.A.shape
        self.training_X = self.training_X[0:self.A.shape[1] * 40]
        self.training_Y = self.training_Y[0:self.A.shape[1] * 40]
        self.get_metadata()

    def get_metadata(self):
        self.data_dim = self.A.shape[0]
        self.m = self.A.shape[1]
        self.n_nodes_each_dim = int(math.pow(self.A.shape[1] * 1., 1. / self.data_dim) + 0.5)
        print 'self.n_nodes_each_dim =', self.n_nodes_each_dim
        self.node_coordinates = self.A[-1, 0:self.n_nodes_each_dim]
        print self.node_coordinates

    def get_L(self):
        L_indices_list = [[], []]
        self.get_L_bfs(0, 0, 0, L_indices_list)
        i0 = np.array(L_indices_list[0], dtype=np_idx_type())
        i1 = np.array(L_indices_list[1], dtype=np_idx_type())
        L_i0 = np.concatenate([i0, i1])
        L_i1 = np.concatenate([i1, i0])
        values = np.ones_like(L_i0, dtype=np_idx_type())
        E = csc_matrix((values, (L_i0, L_i1)), dtype=np_idx_type())
        ones = np.ones(shape=[self.A.shape[1]], dtype=np_data_type())
        tmp = E.dot(ones)
        sum = tmp.dot(ones)

        indices = np.arange(0, self.A.shape[1], dtype=np_idx_type())
        tmp = csc_matrix((tmp, (indices, indices)), dtype=tmp.dtype)
        self.L = (tmp - E) / sum * 2.

    def getW(self, X):
        indices_list = [[],[]]
        for i in range(self.data_dim):
            one_dim_data = X[i]
            indices = np.searchsorted(self.node_coordinates, one_dim_data, side='right')
            indices = np.clip(indices, a_min=1,a_max=self.node_coordinates.shape[0] - 1)
            indices_list[0].append(indices-1)
            indices_list[1].append(indices)

        indices_offsets = np.zeros(shape=[X.shape[1]],dtype=np_idx_type())
        node_indices_list = []
        self.cal_node_indices_bfs(indices_list, 0, indices_offsets, node_indices_list)

        dists_list = []
        for i in range(len(node_indices_list)):
            indices = node_indices_list[i]
            # print '----indices.shape =', indices.shape, 'X.shape =', self.X.shape, 'A.shape =', self.A.shape
            offsets = self.A[:, indices] - X
            dists = np.linalg.norm(offsets, axis=0)
            dists_list.append(dists)

        dists_inverse_array = 1. / (np.array(dists_list, dtype=np_data_type()).transpose() + 1e-8)
        print '------dists_array.shape =', dists_inverse_array.shape
        sum_along_rows = np.sum(dists_inverse_array, axis=1)
        weights = dists_inverse_array / np.reshape(sum_along_rows, [sum_along_rows.shape[0], 1])
        weights_list = []
        for i in range(weights.shape[1]):
            weights_list.append(weights[:,i])

        X_indices = np.arange(0, X.shape[1], dtype=np_idx_type())
        X_indices_list = [X_indices] * len(dists_list)
        all_node_indices = np.concatenate(node_indices_list)
        all_X_indices = np.concatenate(X_indices_list)
        all_weights = np.concatenate(weights_list)
        W = csc_matrix((all_weights,(all_node_indices, all_X_indices)), shape=[self.A.shape[1], X.shape[1]], dtype=np_data_type())
        return W, node_indices_list

    def cal_node_indices_bfs(self, indices_list, dim, offsets, node_indices_list):
        offsets_1 = offsets * self.n_nodes_each_dim + indices_list[0][dim]
        offsets_2 = offsets * self.n_nodes_each_dim + indices_list[1][dim]
        if dim == self.data_dim - 1:
            node_indices_list.append(offsets_1)
            node_indices_list.append(offsets_2)
        else:
            self.cal_node_indices_bfs(indices_list, dim + 1, offsets_1, node_indices_list)
            self.cal_node_indices_bfs(indices_list, dim + 1, offsets_2, node_indices_list)

    def get_L_bfs(self, dim, base_1, base_2, res):
        if dim == self.data_dim - 1:
            if base_1 == base_2:
                for i in range(self.n_nodes_each_dim - 1):
                    res[0].append(base_1 * self.n_nodes_each_dim + i)
                    res[1].append(base_1 * self.n_nodes_each_dim + i + 1)
            else:
                for i in range(self.n_nodes_each_dim):
                    res[0].append(base_1 * self.n_nodes_each_dim + i)
                    res[1].append(base_2 * self.n_nodes_each_dim + i)
        else:
            if base_1 == base_2:
                for i in range(self.n_nodes_each_dim):
                    next_base_1 = base_1 * self.n_nodes_each_dim + i
                    self.get_L_bfs(dim + 1, next_base_1, next_base_1, res)
                for i in range(self.n_nodes_each_dim - 1):
                    next_base_1 = base_1 * self.n_nodes_each_dim + i
                    next_base_2 = base_1 * self.n_nodes_each_dim + i + 1
                    self.get_L_bfs(dim + 1, next_base_1, next_base_2, res)
            else:
                for i in range(self.n_nodes_each_dim):
                    next_base_1 = base_1 * self.n_nodes_each_dim + i
                    next_base_2 = base_2 * self.n_nodes_each_dim + i
                    self.get_L_bfs(dim + 1, next_base_1, next_base_2, res)

    def cal_B(self):
        # self.alpha = 0.01
        self.alpha = 1.
        left = self.training_W.dot(self.training_Y.transpose()) / self.m
        print '--------------', type(left), 'left.shape =', left.shape
        tmp = self.training_W.dot(self.training_W.transpose()) / self.m + self.alpha * self.L
        tmp = tmp.transpose()
        print '--------------', type(tmp), 'tmp.shape =', tmp.shape

        self.B = (spsolve(A=tmp, b=left)).transpose()
        # Bs = []
        # for i in range(left.shape[1]):
        #     B, _, _, _, _, _, _, _, _, _, = lsqr(A=tmp,b=left[:,i])
        #     Bs.append(B)
        # self.B = np.array(Bs).transpose()
        # self.B = B.transpose()
        print '--------------', type(self.B), 'B.shape =', self.B.shape

    def train(self, lattice_dir):
        self.load_training_data(lattice_dir)

        start = time.time()
        self.get_L()
        self.training_W, _ = self.getW(self.training_X)
        self.cal_B()
        end = time.time()
        print 'time =', end - start

    def fit(self, X):
        W, node_indices_list = self.getW(X.transpose())
        W = W.transpose() # sparse n x m matrix
        # print '--------W.shape =', W.shape, 'B.shape =', self.B.shape
        print '--------type(W) =', type(W)
        return W.dot(self.B.transpose()), node_indices_list


    # def cal_B(self):
    #     self.alpha = 0.01
    #     left = self.training_W.dot(self.training_Y.transpose()).transpose() / self.m
    #     print '--------------', type(left), 'left.shape =', left.shape
    #     tmp = self.training_W.dot(self.training_W.transpose()) / self.m + self.alpha * self.L
    #     print '--------------', type(tmp), 'tmp.shape =', tmp.shape
    #     right = inv(tmp)
    #     print '--------------', type(right), 'right.shape =', right.shape
    #
    #     # c = np.linalg.inv(np.linalg.cholesky(tmp))
    #     # right = np.dot(c.T, c)
    #
    #
    #     # left = self.training_Y.dot(self.training_W.transpose()) / self.m
    #     self.B = (right.dot(left.transpose())).transpose()
    #     print '--------------', type(self.B), 'B.shape =', self.B.shape





    def save(self, model_dir):
        A_path = os.path.join(model_dir, 'A.npy')
        lattice_nodes_path = os.path.join(model_dir, 'lattice_nodes.npy')
        B_path = os.path.join(model_dir, 'B.npy')
        training_X_path = os.path.join(model_dir, 'training_X.npy')
        training_Y_path = os.path.join(model_dir, 'training_Y.npy')
        training_W_path = os.path.join(model_dir, 'training_W.npz')
        L_path = os.path.join(model_dir, 'L.npz')
        np.save(A_path, self.A)
        np.save(lattice_nodes_path, self.A.transpose())
        np.save(B_path, self.B)
        np.save(training_X_path, self.training_X)
        np.save(training_Y_path, self.training_Y)
        save_npz(training_W_path, self.training_W)
        save_npz(L_path, self.L)

    def load(self, model_dir):
        A_path = os.path.join(model_dir, 'A.npy')
        B_path = os.path.join(model_dir, 'B.npy')
        training_X_path = os.path.join(model_dir, 'training_X.npy')
        training_Y_path = os.path.join(model_dir, 'training_Y.npy')
        training_W_path = os.path.join(model_dir, 'training_W.npz')
        L_path = os.path.join(model_dir, 'L.npz')

        self.A = np.load(A_path)
        self.B = np.load(B_path)
        # print 'B.shape =', self.B.shape
        # print 'B.type =', self.B.dtype
        # print self.B[0, 0:10]
        # print self.A[0, 0:10]
        # print self.A[2, 0:10]

        self.training_X = np.load(training_X_path)
        self.training_Y = np.load(training_Y_path)
        self.training_W = load_npz(training_W_path)
        self.L = load_npz(L_path)
        self.get_metadata()


    def load_for_fit_only(self, model_dir):
        A_path = os.path.join(model_dir, 'A.npy')
        B_path = os.path.join(model_dir, 'B.npy')
        training_X_path = os.path.join(model_dir, 'training_X.npy')
        training_Y_path = os.path.join(model_dir, 'training_Y.npy')
        training_W_path = os.path.join(model_dir, 'training_W.npz')
        L_path = os.path.join(model_dir, 'L.npz')
        self.A = np.load(A_path)
        print 'A.shape =', self.A.shape
        self.B = np.load(B_path)
        print 'B.shape =', self.B.shape
        self.L = load_npz(L_path)
        print 'L.shape =', self.L.shape
        self.get_metadata()




if __name__ == '__main__':
    home_dir = '/home/pfl/LearnedIndex/4d_uniform'
    Config(home_dir)
    n_lattices_each_dim = Config().n_nodes_each_dim
    model_dir = os.path.join(Config().models_dir, 'lattice_regression')
    model_dir = os.path.join(model_dir, str(n_lattices_each_dim))
    FileViewer.detect_and_create_dir(model_dir)
    lattice_dir = os.path.join(Config().data_dir, 'lattice')
    lattice_dir = os.path.join(lattice_dir, str(n_lattices_each_dim))
    lat_reg = LatticeRegression()
    lat_reg.train(lattice_dir)
    # lat_reg.save(model_dir)

    # lat_reg.load(model_dir)
    # B = lat_reg.B.transpose()
    # print B[200:210]
    # lat_reg.save(model_dir)

    # lat_reg.load(model_dir)
    # # lat_reg.A = np.load("/home/pfl/DB_ML/LearnedIndex/3d_synthetic/data/lattice/50/lattice_nodes.npy").transpose()
    # # print lat_reg.A.shape
    # # print lat_reg.A[-1,0:10]
    # lat_reg.save(model_dir)

