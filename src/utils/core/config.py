"""Library-wise configurations."""

import errno
import os
import sys
sys.path.append("../../../")
from src.utils import FileViewer


class Config(object):
    """Configuration class."""

    class __Singleton(object):
        """Singleton design pattern."""

        def __init__(self, home_dir, models_dir, data_dir, logs_dir):
            """Constructor.
            Parameters
            ----------
            models_dir : string, optional (default='models/')
                directory path to store trained models.
            data_dir : string, optional (default='data/')
                directory path to store model generated data.
            logs_dir : string, optional (default='logs/')
                directory path to store yadlt and tensorflow logs.
            """
            self.home_dir = home_dir
            self.models_dir = os.path.join(self.home_dir, models_dir)
            self.data_dir = os.path.join(self.home_dir, data_dir)
            # self.logs_dir = os.path.join(self.home_dir, logs_dir)
            FileViewer.detect_and_create_dir(self.home_dir)
            FileViewer.detect_and_create_dir(self.models_dir)
            FileViewer.detect_and_create_dir(self.data_dir)
            # FileViewer.detect_and_create_dir(self.logs_dir)

            self.cell_params_path = "cell_params.npy"
            self.data_dim = 4 # The dimension of spatial data
            self.sigma = 100
            self.n_nodes_each_dim = 50 # number of nodes in each dim
            self.max_value = 10000 # max value
            self.n_parts_each_dim = 32 # number of parts in dim 0
            self.n_piecewise_models = 1024 # number of parts in other dims
            self.eta = 0.01
            self.page_size = 60
            self.min_value = 0
            self.static_data_name = "data_1.npy" # static data path
            self.data_to_insert_name = "data_2.npy" # data to insert
            self.lr = 1e-1
            self.query_range_path = "query_ranges.txt"
            print '---------Config is initilized and final----------'

    instance = None

    def __new__(cls, home_dir=os.path.join(os.path.expanduser("~"), '.pfl'), models_dir='models/',
                data_dir='data/',
                logs_dir='logs/'):
        """Return singleton instance."""
        if not Config.instance:
            Config.instance = Config.__Singleton(home_dir, models_dir, data_dir, logs_dir)
        return Config.instance

    def __getattr__(self, name):
        """Get singleton instance's attribute."""
        return getattr(self.instance, name)

    def __setattr__(self, name):
        """Set singleton instance's attribute."""
        return setattr(self.instance, name)
