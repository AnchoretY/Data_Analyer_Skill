## 基础操作
### 1.基本数据类型
&emsp;&emsp;Tensor文pytorch中的基本数据类型，可以看做是包含单一数据类型元素的多维矩阵，在使用时其内部元素数据类型有7中CPU Tensor和8中GPU Tensor可以进行选择，在使用时可以根据网络模型所需要的精度与显存容量来进行选择，其中16位半浮点精读是专门为GPU上运行模型设计的，以尽可能的减少GPU显存占用。


&emsp;&emsp;在pytorch中默认数据类型为torch.FloatTensor,即使用torch.Tensor声明Tensor生成的变量即为torch.FloatTensor。

### 2.Tensor创建
&emsp;&emsp;







### 1.稀疏张量
&emsp;&emsp;pytorch支持使用coo格式的稀疏张量,与scipy中不同的是pytorch中分别输入行索引和列索引不同，pytorch将行索引和列索引组成一个二维的tensor进行输入。具体使用方式如下所示：

```python
i = torch.LongTensor([[0, 1, 1],
                    [2, 0, 2]])
v = torch.FloatTensor([3, 4, 5])
sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size([2,3]))
print(sparse_tensor.to_dense())

output:
  tensor([[0., 0., 3.],
          [4., 0., 5.]])
```
##### scipy sparse metric转换成torch sparse tensor
&emsp;&emsp;pytorch中`不能直接将sparse metric转化成sparse tensor，只能通过重新拆解索引、数据来重新构成sparse tensor`，具体的使用方式如下所示。
```python
import scipy.sparse as sp
i = np.array([1,0])
j = np.array([0,1])
d = np.array([4,5])

sparse_matric = sp.coo_matrix((d,(i,j)),(2,2))    # 创建稀疏矩阵

# 组成tensor索引，tensor数据
index = torch.from_numpy(
    np.array([
        sparse_matric.row,sparse_matric.col]
)).long()
data = torch.from_numpy(
    np.array(sparse_matric.data)
)

sparse_tensor = torch.sparse.FloatTensor(index,data,torch.Size([2,2]))  # 利用前面的索引和数据重新构成稀疏tensor
print("sparse tensor: {}".format(sparse_tensor.to_dense()))

output:
  sparse tensor: tensor([[0, 5],
        [4, 0]])
```





### 2.tensor使用切片操作
&emsp;&emsp;pytorch中的tensor和list类似也可以使用切片进行操作，下面为使用切片操作进行一些简单操作的实例。
```python
data = torch.randn((2,3))
print("origin data:{}".format(data))

print("col select:{}".format(data[:,1]))    # 切片选择张量
print("row select:{}".format(data[1,:]))

data[:,1] = 1    # 使用切片操作修改张量内容
print("changed data:{}".format(data))

output:
  origin data:tensor([[ 0.0063,  1.0000, -0.7999],
          [ 0.1557,  1.0000,  1.6829]])
  col select:tensor([-0.5666, -0.8882])
  row select:tensor([ 0.1557, -0.8882,  1.6829])
  changed data:tensor([[ 0.0063,  1.0000, -0.7999],
          [ 0.1557,  1.0000,  1.6829]])
```

### 3. tensor使用布尔值索引进行操作
&emsp;&emsp;pytorch实现了和list、array一致的采用布尔值对tensor进行操作的接口，只是该接口在使用bool进行索引时，要使用的是内容为bool的tensor来进行索引，而不是list，该接口将在未来对tensor的操作中起到非常大的作用。下面是使用布尔值索引进行操作的实例：
```python
data = torch.tensor([
    [1,2,3],
    [4,5,6]
])

print("origin data: {}".format(data))
print("data==2 data: {}".format(data==2))
data[data==2]=-1    # 使用布尔值索引修改张量内容
print("changed data: {}".format(data))

output:
  origin data: tensor([[1, 2, 3],
          [4, 5, 6]])
  data==2 data: tensor([[False,  True, False],
          [False, False, False]])
  changed data: tensor([[ 1, -1,  3],
          [ 4,  5,  6]])
```

### 4. 初始化函数


