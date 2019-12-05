# Python Usage


*  \__bases__ : 查看父类
e.g. Child.__bases__
* \__doc__ : 该属性不会被继承
* 如果子类没有定义自己的初始化函数，父类的初始化函数会被默认调用；但是如果要实例化子类的对象，则只能传入父类的初始化函数对应的参数。
* 如果子类定义了自己的初始化函数，而在子类中没有显示调用父类的初始化函数，则父类的属性不会被初始化。
* 如果子类定义了自己的初始化函数，在子类中显示调用父类，子类和父类的属性都会被初始化。 

## 2018-10-28
```
__init__.py的主要作用是：
1. package的标识，不能删除
2. 定义__all__用力模糊导入
3. 编写python代码（不建议）

enumerate(sequence, [start=0])

np.flatten()
np.in1d()
#Returns a boolean array the same length as ar1 that is True where an element of ar1 is in ar2 and False otherwise.

#python的切片：
a[i,j] #表示复制a[i]到a[j-1]，以生成新的list对象
#i缺省时，默认为0，即a[:3]相当与a[0:3]
#j缺省时，默认为len(alist)，即a[1:]相当于a[1:10]
#当i,j都缺省时，a[:]就相当于完整复制一份a

b[i:j:s] #s表示步进，缺省为1
#当s<0时，i缺省时，默认为-1，j缺省时默认为-len(a)-1
a[::-1] 

np.setdiff1d(ar1,ar2,assume_unique=False)
#Find the set difference of two arrays
#Return the sorted,unique value in ar1 that are not in ar2
np.intersect1d(ar1,ar2,assume_unique=False,return_indices)
#Find the intersection of two arrays.
#Return the sorted,unique values that are in both of the input arrays

np.ravel()   #返回的是视图
np.flatten() #返回的是拷贝

```

## OS
```
    1. 当前路径及当前路径下的文件
    os.getcwd() 
    # 列举目录下的所有文件，返回的是列表类型
    os.listdir(path) 
    # 列举目录下的所有文件。

    2.绝对路径
    os.path.abspath(path)
    #  返回path的绝对路径

    3. 查看路径的文件夹部分和文件名部分
    os.path.split(path)
    # 将路径分解为(文件夹，文件名)，返回的是元组类型。
    os.path.join(path1,path2,...)
    # 将path进行组合，若其中有绝对路径，则之前的path被删除

    os.path.dirname(path)
    # 返回目录名
    os.path.basename(path)
    # 返回文件名

    os.path.getmtime(path):最后修改时间
    os.path.getatime(path):最后访问时间
    os.path.getctime(path):最后创建时间

    os.path.getsize(path):文件大小

    os.path.exists(path):文件是否存在

    os中定义了一组文件，路径在不同操作系统中表现形式仓鼠
    os.sep : '\\'    
    os.extsep:'.'   后缀
    os.pathsep: ';'  路径
    os.linesep: '\r\n'  
```

## 2018-11-13
```python
dict.setdefault(key, default=None)
#如果键不存在于字典中，将会添加键并将值设为默认值。

#随机取样
sample_number = int(total_number * sample_ratio)
sample_list = random.sample(total_list, sample_number)
unsample_list = [ w for w in total_list if w not in sample_list ]

```

## 2018-12-14
```python
#built-in function
all(iterable)
any(iterable)
bin(x)
enumerate(iterable,start=0)
chr(num)  返回ascii码为num的字符
delatrr(object, 'name')   删除属性 
filter(function, iterable)

```

## 2019-12-3
```python 
def template():
    def __init__():
        pass 
    '''
        with xxx as xxx:
        with要求对象必须有__enter__()&__exit__()方法，__enter__()返回的值赋给as后面的值
    '''
    def __enter__():
        pass 

    def __exit__():
        pass

    '''
        __str__方法只适用于print(), 不适用于shell中的直接输出。
    '''
    def __str__(self,):
        pass 

    def __repr__(self,):
        pass 

```

### 相对引用
* python某个module中使用了相对引用，同时这个module的__name__属性又是__main__会报错问题。如果一个python文件是__main__，那么相对引用会被解析成top level module，而不关注这个module是否存在这个文件系统中。

## 2019-12-4
```python
time.strftime("%m-%d-%H:%M:%S")
```

* 文件目录
```python
os.mkdirs() #建立多级目录
```

* set
```python
set.difference()
```

* itertools

* setattr()/getattr()
```python
setattr(self, name, )
```

* glob 模块
```
glob.glob("")
#只用到三个匹配符，"*", "?", "[]" "*"匹配0个或多个字符，"?"匹配单个字符： "[]"匹配指定范围内字符。
```


