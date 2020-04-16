### 1.稀疏张量
&emsp;&emsp;




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


