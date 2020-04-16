### 1. coo稀疏矩阵
&emsp;&emsp;是一种坐标形式的稀疏矩阵。`采用三个数组row、col和data保存非零元素的信息，这三个数组的长度相同，row保存元素的行，
col保存元素的列，data保存元素的值`。存储的主要`优点`是`灵活、简单，仅存储非零元素以及每个非零元素的坐标`。但是COO`不支持元素的存取和
增删，一旦创建之后，除了将之转换成其它格式的矩阵，几乎无法对其做任何操作和矩阵运算`。
![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.idx2bm86fe.png)
&emsp;&emsp;创建方式如下：
```python
import scipy.sparse as sp

data = np.ones((2),dtype=int)
row = np.array([0,0])
col = np.array([1,2])

sparse_data = sp.coo_matrix((a,(row,col)),shape=(3,3))
print(sparse_data.toarray())

output:
  array([[0., 1., 1.],
       [0., 0., 0.],
       [0., 0., 0.]])
```
##### 获取数据索引
&emsp;&emsp;coo稀疏矩阵可以直接使用row、col、data访问其中的索引与数据.
```python
print("row index:{}".format(sparse_data.row))
print("col index:{}".format(sparse_data.col))
print("data:{}".format(sparse_data.data))

output:
  row index:[0 0]
  col index:[1 2]
  data:[1 1]
```






##### 参考文献
- https://blog.csdn.net/pipisorry/article/details/41762945
