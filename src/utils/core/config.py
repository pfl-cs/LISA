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

        def __init__(self):
            """Constructor.
            Parameters
            ----------
            models_dir : string, optional (default='models/')
                directory path to store trained models.
            data_dir : string, optional (default='data/')
                directory path to store model generated data.
            """
            data_name = '4d_uniform'
            self.home_dir = os.path.join(os.path.expanduser("~"), os.path.join('workspace/LISA', data_name))
            self.models_dir = os.path.join(self.home_dir, 'models')
            self.data_dir = os.path.join(self.home_dir, 'data')
            # self.logs_dir = os.path.join(self.home_dir, logs_dir)
            FileViewer.detect_and_create_dir(self.home_dir)
            FileViewer.detect_and_create_dir(self.models_dir)
            FileViewer.detect_and_create_dir(self.data_dir)
            # FileViewer.detect_and_create_dir(self.logs_dir)


            self.query_range_path = os.path.join(self.data_dir, "query_ranges.qr")
            self.static_data_name = "data_0.npy"  # static data path
            self.data_to_insert_name = "data_2.npy"  # data to insert
            self.data_to_delete_name = "data_3.npy"  # data to delete
            self.cell_params_path = "cell_params.npy"


            # params for bulk loading
            self.data_dim = 4 # The dimension of spatial data
            self.sigma = 100
            self.max_value = 10000 # max value
            self.T_each_dim = 32 # number of parts in each dim
            self.n_piecewise_models = 1024
            self.eta = 0.01
            self.page_size = 60
            self.min_value = 0

            self.lr = 1e-1


            self.tau = 50 # number of nodes in each dim


            print '---------Config is initilized----------'

    instance = None

    def __new__(cls):
        """Return singleton instance."""
        if not Config.instance:
            Config.instance = Config.__Singleton()
        return Config.instance

    def __getattr__(self, name):
        """Get singleton instance's attribute."""
        return getattr(self.instance, name)

    def __setattr__(self, name):
        """Set singleton instance's attribute."""
        return setattr(self.instance, name)
