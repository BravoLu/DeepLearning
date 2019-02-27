import time


''' 装饰器本质上是一个Python函数，可以让其他函数再不需要做任何代码变动的前提下增加额外的功能，
    装饰器的返回值也是一个函数对象。它经常用于有切面需求的场景，比如
    插入日志，性能测试，事务处理，缓存，权限校验等场景。装饰器是解决这类问题的绝佳设计，有了装饰器，我们就可以
    抽离出大量与函数功能本身无关的雷同代码并继续重用。
    
'''
def deco(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        endTime = time.time()
        msecs = (endTime - startTime)*1000
        print("time is %d ms" %msecs)
    return wrapper


@deco
def func(a,b):
    print("hello，here is a func for add :")
    time.sleep(1)
    print("result is %d" %(a+b))

@deco
def func2(a,b,c):
    print("hello，here is a func for add :")
    time.sleep(1)
    print("result is %d" %(a+b+c))


if __name__ == '__main__':
    f = func
    func2(3,4,5)
    f(3,4)
    #func()