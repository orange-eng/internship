# Coding Standard
Learn something about coding standard about Python in Baidu.


- 1.只能使用一下基本类型作为默认参数：int,bool,float,string,None
- 2.protected成员使用单下划线前缀，private成员使用双下划线前缀
- 3.如果一个类没有基类，那么必须继承自object类
- 4.使用%或是format格式化字符串

| 格式化字符串 | 说明                                       |
| --------   | -----:                                     |
| %c        | 转换成字符（ASCII 码值，或者长度为一的字符串） |   
| %r        | 优先用repr()函数进行字符串转换                |
| %s        | 优先用str()函数进行字符串转换                 |
| %d %i     | 转化成有符号十进制数字                        |
| %u        | 转化成无符号十进制数字                        |
| %o        | 转化成无符号八进制数字                        |

注意：
- str()得到的字符串是面向用户的，具有较好的可读性
- repr()得到的字符串是面向机器的

- 5.不能使用+=来拼接字符串列表，而要使用join
- 6.所有module必须可以导入。如果要执行主程序，必须检查__name__=="__main__"


#### example_5:
```python
str = "+"
seq = ("a", "b", "c"); # 字符串序列
print(str.join(seq))
#结果：a+b+c
list=['1','2','3','4','5']
print(''.join(list)) #用空字符串连接
#结果：12345
seq = {'hello':1,'world':2,'boy':3,'girl':4}
print('-'.join(seq))        
#字典只对键进行连接
#结果：hello-world-boy-girl
```

#### example_2:
```python
class ClassDef(object):
    def __init__(self):
        # public
        self.name = "class_def"
        # private
        self.__age = 29
        # protected
        self._sex = "man"
    def fun1(self):
        print("call public function")
    def __fun2(self):
        print("call private function")
    def _fun3(self):
        print("call protected function")

if __name__ == "__main__":
    # 实例化类对象
    class_def = ClassDef()
    # 调用方法
    # ok
    class_def.fun1()
    class_def._ClassDef__fun2()
    class_def._fun3()
    # 访问数据
    print(class_def._ClassDef__age)
    print(class_def._sex)
    print(class_def.name)
    # error
    # class_def.__fun2()
    # print(class_def.__age)
```
