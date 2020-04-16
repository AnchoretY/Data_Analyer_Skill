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
