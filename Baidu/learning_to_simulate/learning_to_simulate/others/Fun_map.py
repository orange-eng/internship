

def square(x):
    return x*x
l = map(square, [1,2,3])
print(l)
print("返回的map对象类型：",type(l))
print("强制转换列表后：",list(l))
