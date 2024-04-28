### HDF5 简介

!!!note
    from https://www.cnblogs.com/-wenli/p/14020264.html

HDF（Hierarchical Data Format）指一种为存储和处理大容量科学数据设计的文件格式及相应库文件。HDF 最早由美国国家超级计算应用中心 NCSA 开发，目前在非盈利组织 HDF 小组维护下继续发展。当前流行的版本是 HDF5。HDF5 拥有一系列的优异特性，使其特别适合进行大量科学数据的存储和操作，如它支持非常多的数据类型，灵活，通用，跨平台，可扩展，高效的 I/O 性能，支持几乎无限量（高达 EB）的单文件存储等，详见其官方介绍：[https://support.hdfgroup.org/HDF5/](https://link.jianshu.com/?t=https%3A%2F%2Fsupport.hdfgroup.org%2FHDF5%2F) 。

## HDF5 结构

HDF5 文件一般以 .h5 或者 .hdf5 作为后缀名，需要专门的软件才能打开预览文件的内容。HDF5 文件结构中有 2 primary objects: Groups 和 Datasets。

*   Groups 就类似于文件夹，每个 HDF5 文件其实就是根目录 (root) group`'/'，可以看成目录的容器，其中可以包含一个或多个 dataset 及其它的 group`。
*   Datasets 类似于 [NumPy](https://link.zhihu.com/?target=https%3A//numpy.org/) 中的数组 array，可以当作数组的数据集合 。

每个 dataset 可以分成两部分: 原始数据 (raw) data values 和 元数据 metadata (a set of data that describes and gives information about other data => raw data)。

```python
+-- Dataset
|   +-- (Raw) Data Values (eg: a 4 x 5 x 6 matrix)
|   +-- Metadata
|   |   +-- Dataspace (eg: Rank = 3, Dimensions = {4, 5, 6})
|   |   +-- Datatype (eg: Integer)
|   |   +-- Properties (eg: Chuncked, Compressed)
|   |   +-- Attributes (eg: attr1 = 32.4, attr2 = "hello", ...)
|
```

从上面的结构中可以看出：

*   Dataspace 给出原始数据的秩 (Rank) 和维度 (dimension)
*   Datatype 给出数据类型
*   Properties 说明该 dataset 的分块储存以及压缩情况

*   Chunked: Better access time for subsets; extendible
*   Chunked & Compressed: Improves storage efficiency, transmission speed

*   Attributes 为该 dataset 的其他自定义属性

整个 HDF5 文件的结构如下所示：

```python
+-- /
|   +-- group_1
|   |   +-- dataset_1_1
|   |   |   +-- attribute_1_1_1
|   |   |   +-- attribute_1_1_2
|   |   |   +-- ...
|   |   |
|   |   +-- dataset_1_2
|   |   |   +-- attribute_1_2_1
|   |   |   +-- attribute_1_2_2
|   |   |   +-- ...
|   |   |
|   |   +-- ...
|   |
|   +-- group_2
|   |   +-- dataset_2_1
|   |   |   +-- attribute_2_1_1
|   |   |   +-- attribute_2_1_2
|   |   |   +-- ...
|   |   |
|   |   +-- dataset_2_2
|   |   |   +-- attribute_2_2_1
|   |   |   +-- attribute_2_2_2
|   |   |   +-- ...
|   |   |
|   |   +-- ...
|   |
|   +-- ...
|

```

一个 HDF5 文件从一个命名为 "/" 的 group 开始，所有的 dataset 和其它 group 都包含在此 group 下，当操作 HDF5 文件时，如果没有显式指定 group 的 dataset 都是默认指 "/" 下的 dataset，另外类似相对文件路径的 group 名字都是相对于 "/" 的。

安装

```python
pip install h5py
```

Python 读写 HDF5 文件

```python
#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# Created by WW on Jan. 26, 2020
# All rights reserved.
#

import h5py
import numpy as np

def main():
    #===========================================================================
    # Create a HDF5 file.
    f = h5py.File("h5py_example.hdf5", "w")    # mode = {'w', 'r', 'a'}

    # Create two groups under root '/'.
    g1 = f.create_group("bar1")
    g2 = f.create_group("bar2")

    # Create a dataset under root '/'.
    d = f.create_dataset("dset", data=np.arange(16).reshape([4, 4]))

    # Add two attributes to dataset 'dset'
    d.attrs["myAttr1"] = [100, 200]
    d.attrs["myAttr2"] = "Hello, world!"

    # Create a group and a dataset under group "bar1".
    c1 = g1.create_group("car1")
    d1 = g1.create_dataset("dset1", data=np.arange(10))

    # Create a group and a dataset under group "bar2".
    c2 = g2.create_group("car2")
    d2 = g2.create_dataset("dset2", data=np.arange(10))

    # Save and exit the file.
    f.close()

    ''' h5py_example.hdf5 file structure
    +-- '/'
    |   +--    group "bar1"
    |   |   +-- group "car1"
    |   |   |   +-- None
    |   |   |   
    |   |   +-- dataset "dset1"
    |   |
    |   +-- group "bar2"
    |   |   +-- group "car2"
    |   |   |   +-- None
    |   |   |
    |   |   +-- dataset "dset2"
    |   |   
    |   +-- dataset "dset"
    |   |   +-- attribute "myAttr1"
    |   |   +-- attribute "myAttr2"
    |   |   
    |   
    '''

    #===========================================================================
    # Read HDF5 file.
    f = h5py.File("h5py_example.hdf5", "r")    # mode = {'w', 'r', 'a'}

    # Print the keys of groups and datasets under '/'.
    print(f.filename, ":")
    print([key for key in f.keys()], "\n")  

    #===================================================
    # Read dataset 'dset' under '/'.
    d = f["dset"]

    # Print the data of 'dset'.
    print(d.name, ":")
    print(d[:])

    # Print the attributes of dataset 'dset'.
    for key in d.attrs.keys():
        print(key, ":", d.attrs[key])

    print()

    #===================================================
    # Read group 'bar1'.
    g = f["bar1"]

    # Print the keys of groups and datasets under group 'bar1'.
    print([key for key in g.keys()])

    # Three methods to print the data of 'dset1'.
    print(f["/bar1/dset1"][:])        # 1. absolute path

    print(f["bar1"]["dset1"][:])    # 2. relative path: file[][]

    print(g['dset1'][:])        # 3. relative path: group[]



    # Delete a database.
    # Notice: the mode should be 'a' when you read a file.
    '''
    del g["dset1"]
    '''

    # Save and exit the file
    f.close()

if __name__ == "__main__":
    main()
```

### 相关代码示例

创建一个 h5py 文件

```python
import h5py
f=h5py.File("myh5py.hdf5","w")
```

创建 dataset

```python
import h5py
f=h5py.File("myh5py.hdf5","w")
#deset1是数据集的name，（20,）代表数据集的shape，i代表的是数据集的元素类型
d1=f.create_dataset("dset1", (20,), 'i')
for key in f.keys():
    print(key)
    print(f[key].name)
    print(f[key].shape)
    print(f[key].value)

输出：
dset1
/dset1
(20,)
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
```

赋值

```python
import h5py
import numpy as np
f=h5py.File("myh5py.hdf5","w")

d1=f.create_dataset("dset1",(20,),'i')
#赋值
d1[...]=np.arange(20)
#或者我们可以直接按照下面的方式创建数据集并赋值
f["dset2"]=np.arange(15)

for key in f.keys():
    print(f[key].name)
    print(f[key].value)

输出：
/dset1
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
/dset2
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
```

创建 group

```python
import h5py
import numpy as np
f=h5py.File("myh5py.hdf5","w")

#创建一个名字为bar的组
g1=f.create_group("bar")

#在bar这个组里面分别创建name为dset1,dset2的数据集并赋值。
g1["dset1"]=np.arange(10)
g1["dset2"]=np.arange(12).reshape((3,4))

for key in g1.keys():
    print(g1[key].name)
    print(g1[key].value)

输出：
/bar/dset1
[0 1 2 3 4 5 6 7 8 9]
/bar/dset2
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
```

删除某个 key 下的数据

```python
# 删除某个key，调用remove
f.remove("bar")
```

最后 pandsa 读取 HDF5 格式文件

```python
import pandas as pd
import numpy as np

# 将mode改成r即可
hdf5 = pd.HDFStore("hello.h5", mode="r")
# 或者
"""
hdfs = pd.read_hdf("hello.h5", key="xxx")
"""
```