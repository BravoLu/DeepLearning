# 110道Python面试题汇总
1. 一行实现1-100之和
```python
    sum(range(1,101))
```
2. 如何在一个函数内部修改全局变量
```python
    a = 5
    def fn():
        global a 
        a = 4
```
3. 字典删除键/合并两个字典
```python
    dic1 = {'1':a}
    dic2 = {'2':b}
    dic1.update(dict2) #合并
    del dic1['2']  #删除键
```
4. list取出重复元素
```python
    list = [1, 1, 2, 3]
    a = set(list)
    list = [i for i in a]
```
5. fun(*args, *kwargs)
*args: 不定数量的非键值对参数, *kwargs: 不定数量的键值对
6. [1, 2, 3, 4, 5],请用map()函数输出[1,4,9,16,25],并用列表推导式提取出大于10的数,最终输出[16,25]
```
    a = [1,2,3,4,5]
    a = map(lambda x:x**2, a)
    a = [i for i in a if i > 10]  
    print(a)
```
7. 正则匹配
```python
    res = re.findall('<div class=".*">(.*?)</div>', strings)
```

8. 排序
```python 
    s = [1, 2, 3, 1, 3, 8, 2]
    s.sort(reverse=True) #reverse从大到小, 和sorted不同
```

9. 用lambda函数实现两个数相乘
```python
    mul = lambda x,y:x*y
    res = mul(x,y)
```

10. 字典根据键从大到小排序
```python
    dict = {'1': x, '2':y}
    sorted(dict.items(), lambda k,v:k, reverse=False)
```

11. filter的用法
```python
    def fn(a):
        return a%2==1 
    a = filter(fn, a)
```

