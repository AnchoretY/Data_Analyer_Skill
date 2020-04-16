### 1. 生成内容为布尔类型的narray
&emsp;&emsp;可以通过使用numpy中的zeros、ones等函数指定dtype来进行实现.
```python
x = np.zeros(10,dtype=np.bool)
print(x)
output:
  array([False, False, False, False, False, False, False, False, False,
       False])
```

### 2. 使用布尔类型的索引来选择元素、修改指定值
&emsp;&emsp;numpy支持使用布尔类型的索引来对元素进行选择，索引值为True的被选中，直接复制即可批量修改。
```python
x = np.zeros(5,dtype=np.bool)
x[2:] = True
print(x)

data = np.arange(5)
print("ori data:{}".format(data))
selected_data = data[x]
print("selected_data:{}".format(out_data))
data[x] = -1
print("changed data:{}".format(data))

output:
  [False False  True  True  True]
  ori data:[0 1 2 3 4]
  selected_data:[2 3 4]
  changed data:[ 0  1 -1 -1 -1]
```
