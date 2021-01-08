# Introduction

LISA is a learned index structure for spatial data, similar in functionality to a R-Tree.
More details about LISA can be found in our [SIGMOD 2020 paper](https://doi.org/10.1145/3318464.3389703).

# Getting Started


### Source code Info
We implement LISA with python 2.7 on the Ubuntu opearting system. You need to install two scientific computing package: numpy and scipy.
The codes can be found [here](https://github.com/pfl-cs/LISA).

### Datasets Info

The input data for building LISA is a numpy ndarray file whose shape is NxD where D and N are the dimension and the number of the input keys, respectively.

We prepared two 4-dimensional datasets for validating our approach. [data_0.npy] (https://pan.zju.edu.cn/share/c34732c0b5cdf3a338521820ef) is used for building LISA; [data_2.npy] (https://pan.zju.edu.cn/share/33333f0ca93113fc0aa3b77d9e) is used for analyzing how LISA and perform if we insert new keys.

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

3) Download [data_0.npy] (https://pan.zju.edu.cn/share/c34732c0b5cdf3a338521820ef) and [data_2.npy] (https://pan.zju.edu.cn/share/33333f0ca93113fc0aa3b77d9e). Put them in the directory $Workspace$/LISA/4d_uniform/data/.

4) cd $Workspace$/LISA/src and run main.py

### More
The LISA is saved in the same directory with the corresponding dataset. If you want to test LISA on other datasets, you need to create a new directory $Dir$ under "$Workspace$/LISA/" and put the dataset into "$Workspace$/LISA/$Dir$/data/".

We will update the [repo] (https://github.com/pfl-cs/LISA) if any modifications are made in the future.


