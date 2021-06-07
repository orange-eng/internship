

# -*- coding:utf-8 -*-
f = open(r'stat_smoking.txt','r')
#a = list(f)
line = f.readline() # 读取第一行
print(line)
tu = eval(line)
#print(tu['ugcid'])
url_list = []
while line:
    txt_data = eval(line)
    print(txt_data['ugcid'])
    url_list.append('https://kg.qq.com/node/play?s=' + txt_data['ugcid'])
    line = f.readline()
print(url_list)
# txt_tables = []
# while line:
#     txt_data = eval(line) # 可将字符串变为元组
#     txt_tables.append(txt_data) # 列表增加
#     line = f.readline() # 读取下一行
# print(txt_tables)

f.close()
