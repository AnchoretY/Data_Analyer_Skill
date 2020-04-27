## 基础操作
### 1.基本数据类型
&emsp;&emsp;Tensor文pytorch中的基本数据类型，可以看做是包含单一数据类型元素的多维矩阵，在使用时其内部元素数据类型有7中CPU Tensor和8中GPU Tensor可以进行选择，在使用时可以根据网络模型所需要的精度与显存容量来进行选择，其中16位半浮点精读是专门为GPU上运行模型设计的，以尽可能的减少GPU显存占用。

| 数据类型 |    CPU Tensor     | GPU Tensor |
| :------: | :---------------: | :--------: |
| 32位浮点 | torch.FloatTensor |  torch.cuda.FloatTensor          |
| 64位浮点 | torch.DoubleTensor| torch.cuda.DoubleTensor |
| 16位半点精度浮点 | N/A | torch.cuda.HalfTensor |
| 8位无符号整型 | torch.ByteTensor | torch.cuda.ByteTensor |
| 8位有符号整型 |  torch.CharTensor  |  torch.cuda.CharTensor  |
| 16有符号整型 | torch.ShortTensor | torch.cuda.ShortTensor |
| 32位有符号整型 | torch.IntTensor | torch.cuda.IntTensor |
| 64为有符号整型 | torch.LongTensor | torch.cuda.LongTensor |


&emsp;&emsp;**在pytorch中默认数据类型为torch.FloatTensor**,即使用torch.Tensor声明Tensor生成的变量即为torch.FloatTensor。

### 2.Tensor创建

&emsp;&emsp;在pytorch中Tensor创建方式有很多，其构造接口与numpy的矩阵创建方法十分类似，如ones()、eye()、zeros()等，下面为常见的Tensor创建方式:

#### 1.torch.Tensor(2,2)

&emsp;&emsp;可以使用torch.Tensor(2,2)命令来**创建指定大小的默认类型Tensor（值随机）**.

~~~python
torch.Tensor(2,2)

output:
  tensor([[-7.8454e+30,  4.5685e-41],
          [ 9.0476e-37,  0.0000e+00]])
~~~

#### 2.torch.IntTensor(2,2)

&emsp;&emsp;使用torch.IntTensor(2,2)来创建**指定类型的Tensor(值随机)**，与上面类似。

#### 3.使用Python的list进行创建

&emsp;&emsp;已有list值时可以直接使用list创建**指定内容的Tensor**

~~~python
torch.Tensor([[1,2],[3,4]])

output:
  tensor([[1., 2.],
        [3., 4.]])
~~~

#### 4.torch.zeros(2,2)

&emsp;&emsp;使用torch.zeros(2,2)可以声明指定大小的全零Tensor。

~~~python
torch.zeros(2,2)
output:
  tensor([[0., 0.],
        [0., 0.]])
  
# 使用dtype指定生成bool型Tensor
torch.zeros(2,2,dtype=torch.bool)
output:
  tensor([[False, False],
        [False, False]])
~~~

#### 5.torch.ones(2,2)

&emsp;&emsp;使用torch.ones(2,2)可以声明指定大小的全一Tensor。

#### 6.torch.eys(3,3)

&emsp;&emsp;使用torch.eye(3,3)可以声明指定大小的对角Tensor（对角线上元素为1，其他位置为0）

~~~python
torch.eye(3,3)
ouput:
  tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])
~~~

#### 7.Torch.randn(2,2)系列函数

&emsp;&emsp;使用torch.randn、torch.randint创建指定大小类型的数据。

> torch.randn(size) :生成指定大小的随机浮点型Tensor（标准正态分布中随机采样）
>
> torch.rand(size):生成指定大小的随机浮点型Tensor（均匀正态分布中随机采样）
>
> torch.randint(low,high,size): 生成指定大小的整型Tensor
>
> torch.rand_like(Tensor):生成与输入矩阵大小相同的随机浮点型Tensor（均匀分布中随机采样）
>
> torch.randn_like(Tensor): 生成与输入矩阵大小相同的随机浮点型Tensor（标准正态分布中采样）
>
> torch.randint_like(Tensor): 生成与输入矩阵大小相同的整型Tensor

~~~python
# 生成指定大小的随机浮点型Tensor
torch.randn(2,2)

# 生成指定大小的随机整型Tensor
torch.randint(low=1,high=10,size=(2,2))
output:
  tensor([[2, 8],
        [7, 4]])
  
# 生成形状相同的随机矩阵
a = torch.Tensor(2)
torch.randn_like(a)
output:
  tensor([ 0.1842, -0.2821])
~~~







## 进阶

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


