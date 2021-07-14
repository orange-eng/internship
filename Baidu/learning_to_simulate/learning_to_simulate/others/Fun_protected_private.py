
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

    string = "hello\will\n"
    print("%s"%string)
    print("%r"%string)