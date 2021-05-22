# Introduction

LISA is a learned index structure for spatial data, similar in functionality to a R-Tree.
More details about LISA can be found in our [SIGMOD 2020 paper](https://doi.org/10.1145/3318464.3389703).

# Getting Started


### Source code Info
We implement LISA with python 2.7 on the Ubuntu opearting system. You need to install two scientific computing package: numpy and scipy.
The codes can be found [here](https://github.com/pfl-cs/LISA).

### Datasets Info

The input data for building LISA is a numpy ndarray file whose shape is NxD where D and N are the dimension and the number of the input keys, respectively.

We prepared three 2-dimensional datasets and three 3-dimensional datasets for validating our approach. [2d_uniform_data_0.npy](https://pan.zju.edu.cn/share/badd090cc3ada34ad1d2ea1fd0) / [3d_uniform_data_0.npy](https://pan.zju.edu.cn/share/5bd48ce4b8390578800673c44b) is used for building LISA; [2d_uniform_data_2.npy](https://pan.zju.edu.cn/share/9408a54a4ffbdd42b50b84f0ea) / [3d_uniform_data_2.npy](https://pan.zju.edu.cn/share/f0b033cc4e78547fc38bcb396d) and [2d_uniform_data_3.npy](https://pan.zju.edu.cn/share/ee998fac3a71d3092d15e96812) / [3d_uniform_data_3.npy](https://pan.zju.edu.cn/share/efbfb6a38aeabc199a2c0a0c41) are used for analyzing how LISA performs if we insert and delete keys, respectively.

### Hardware Info
The details of the hardwares we used for the experiments are shown as follows.

+ Processor: 32  Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
+ L1d cache: 32K
+ L1i cache: 32K
+ L2 cache: 256K
+ L3 cache: 20480K
+ Memory: 128GB
+ Secondary storage: HDD 3.7 TB

### Step by step
1) Install the scientific computing packages:

+ pip install numpy; pip install scipy; pip install scikit-learn;

2) Clone the [repository](https://github.com/pfl-cs/LISA) under a directory $Workspace$.

3) Download [2d_uniform_data_0.npy](https://pan.zju.edu.cn/share/badd090cc3ada34ad1d2ea1fd0), [2d_uniform_data_2.npy](https://pan.zju.edu.cn/share/9408a54a4ffbdd42b50b84f0ea), [2d_uniform_data_3.npy](https://pan.zju.edu.cn/share/1f5fb2c99baf268c0b5a23e288) and [2d_uniform_query_ranges.qr](https://pan.zju.edu.cn/share/ee998fac3a71d3092d15e96812). Put them in the directory $Workspace$/LISA/2d_uniform/data/.
Download [3d_uniform_data_0.npy](https://pan.zju.edu.cn/share/5bd48ce4b8390578800673c44b), [3d_uniform_data_2.npy](https://pan.zju.edu.cn/share/f0b033cc4e78547fc38bcb396d), [3d_uniform_data_3.npy](https://pan.zju.edu.cn/share/efbfb6a38aeabc199a2c0a0c41) and [3d_uniform_query_ranges.qr](https://pan.zju.edu.cn/share/9006594a9dbe45ab90e0fc7ad2). Put them in the directory $Workspace$/LISA/3d_uniform/data/.

4) cd $Workspace$/LISA/src and run main.py

### More
The LISA is saved in the same directory with the corresponding dataset. If you want to test LISA on other datasets, you need to create a new directory $Dir$ under "$Workspace$/LISA/" and put the dataset into "$Workspace$/LISA/$Dir$/data/".

We will update the [repo](https://github.com/pfl-cs/LISA) if any modifications are made in the future.


