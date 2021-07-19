def text_create(name, msg):
    current_path = "dataset\\" # 新创建的txt文件的存放路径
    full_path = current_path + name + '.txt' # 也可以创建一个.doc的word文档
    file = open(full_path, 'w')
    file.write(msg) #msg也就是下面的Hello world!
    file.close()

text_create('mytxtfile', 'Hello world!')

#with open("douban.txt","w") as f:
#        f.write("这是个测试！")

