# -*- coding=utf-8 -*-
# @Time    : 2020/12/18 19:24
# @Author  : lhys
# @FileName: baidu.py

import requests
import urllib.parse as up
import json
import time
import os

major_url = 'https://image.baidu.com/search/index?'
headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'}
def pic_spider(kw, page = 10, file_path = os.getcwd()):
    path = os.path.join(file_path, kw)
    if not os.path.exists(path):
        os.mkdir(path)
    if kw != '':
        for num in range(page):
            data = {
                "tn": "resultjson_com",
                "logid": "11587207680030063767",
                "ipn": "rj",
                "ct": "201326592",
                "is": "",
                "fp": "result",
                "queryWord": kw,
                "cl": "2",
                "lm": "-1",
                "ie": "utf-8",
                "oe": "utf-8",
                "adpicid": "",
                "st": "-1",
                "z": "",
                "ic": "0",
                "hd": "",
                "latest": "",
                "copyright": "",
                "word": kw,
                "s": "",
                "se": "",
                "tab": "",
                "width": "",
                "height": "",
                "face": "0",
                "istype": "2",
                "qc": "",
                "nc": "1",
                "fr": "",
                "expermode": "",
                "force": "",
                "pn": num*30,
                "rn": "30",
                "gsm": oct(num*30),
                "1602481599433": ""
            }
            url = major_url + up.urlencode(data)
            i = 0
            pic_list = []
            while i < 5:
                try:
                    pic_list = requests.get(url=url, headers=headers).json().get('data')
                    break
                except:
                    print('网络不好，正在重试...')
                    i += 1
                    time.sleep(1.3)
            for pic in pic_list:
                url = pic.get('thumbURL', '') # 有的没有图片链接，就设置成空
                if url == '':
                    continue
                name = pic.get('fromPageTitleEnc')
                for char in ['?', '\\', '/', '*', '"', '|', ':', '<', '>']:
                    name = name.replace(char, '') # 将所有不能出现在文件名中的字符去除掉
                type = pic.get('type', 'jpg') # 找到图片的类型，若没有找到，默认为 jpg
                pic_path = (os.path.join(path, '%s.%s') % (name, type))
                print(name, '已完成下载')
                if not os.path.exists(pic_path):
                    with open(pic_path, 'wb') as f:
                        f.write(requests.get(url = url, headers = headers).content)
pic_spider('电子红包')
