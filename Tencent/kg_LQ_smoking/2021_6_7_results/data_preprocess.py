#!/usr/bin/python3
# -*- coding: utf-8 -*- 
# @author: yubinnzeng
# @license: (C) Copyright 2021-2021, TME Inc.
# @contact: yubinnzeng@tencent.com
# @file: data_preprocess.py
# @time: 2021/4/22 5:36 下午
import glob

import pandas as pd
from selenium import webdriver
import os
import cv2
import wget

import requests
 




def read_urls_from_excel(path):
    data_frame = pd.read_excel(path)
    video_urls = data_frame['链接'].values
    return video_urls

def read_urls_from_txt(path="instrument_cases.txt"):
    video_urls = []
    with open(path, 'r') as f:
        for line in f:
            video_urls.append(line.strip())
    return video_urls

def download_from_urls(video_url,num):
    """
    download video from kg website as tmp.mp4
    """
    output_vid_name = "tmp.mp4"
    option = webdriver.ChromeOptions()
    option.add_argument("headless")
    driver = webdriver.Chrome(chrome_options=option)

    try:
        print("Downloading: ", video_url)

        driver.get(video_url)  # 打开url
        value_element = driver.find_element_by_id('player')
        mp4file = value_element.get_attribute("src")
        # print("mp4:",mp4file)
        # print("value_element:",value_element)
        file_name = wget.download(mp4file,out= "{}.mp4".format(num))
        # print(file_name)
        # cmd = f"wget -q '{mp4file}' -O {output_vid_name}"
        # print("CMD: ", cmd)
        # ret = os.system(cmd)

        # if ret==0:
        #     print("Downloaded.")
    except Exception as e:
        print(e)

    driver.quit()

    return output_vid_name


def vid2img(video_name="tmp.mp4"):
    # 保存图片的临时路径
    save_path = "video_image"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        imgs_to_remove = glob.glob(os.path.join(save_path, "*"))
        for img_to_remove in imgs_to_remove:
            os.remove(img_to_remove)

    try:
        fps, max_frames = get_video_info(video_name)
        duration = max_frames // fps

        if duration < 5:
            fps, max_frames = 2, 10
        elif duration < 20:
            fps, max_frames = 0.5, 10
        elif duration < 60:
            fps, max_frames = 0.2, 12
        elif duration < 180:
            fps, max_frames = 0.1, 20
        else:
            fps, max_frames = 0.1, 20


        # CommonCmd = f'ffmpeg -hide_banner -loglevel error -y -i {video_name}  -vf fps={fps} -vframes {max_frames} {save_path}//%d.png'
        #os.system(CommonCmd)
        return save_path

    except Exception as e:
        os.rmdir(save_path)
        print("err")
        return "err"


def get_video_info(file):
    video = cv2.VideoCapture(file)
    nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps, nframes


if __name__ == '__main__':
    # BADCASE_EXCEL = "/Users/zengyubin/PycharmProjects/docker_low_quality_kg/badcases_20210526.xlsx"

    # urls = read_urls_from_excel(BADCASE_EXCEL)
    # print("All cases: \n", urls)
    # for url in urls:
    #     vid_name = download_from_urls(url)

    #     img_folder = vid2img(vid_name)

    #     print(glob.glob(os.path.join(img_folder, "*.png")))

#-----------------------------------------------------------------------------
    f = open(r'stat_smoking.txt','r')
    #a = list(f)
    line = f.readline() # 读取第一行
    print(line)
    tu = eval(line)
    #print(tu['ugcid'])
    urls_list = []
    while line:
        txt_data = eval(line)
        #print(txt_data['ugcid'])
        urls_list.append('https://kg.qq.com/node/play?s=' + txt_data['ugcid'])
        line = f.readline()
    #print(url_list)

    # urls_list = [
    # 'https://kg.qq.com/node/play?s=11532513_1621393818_369',
    # 'https://kg.qq.com/node/play?s=8640926_1621651347_630',
    # 'https://kg.qq.com/node/play?s=3232034_1621554606_255',
    # 'https://kg.qq.com/node/play?s=30803152_1621203249_557',
    # 'https://kg.qq.com/node/play?s=5542928_1621770525_41',
    # 'https://kg.qq.com/node/play?s=17012679_1621721116_407'
    # ]
    for i in range(len(urls_list)):
        vid_name = download_from_urls(urls_list[i],i)
    #img_folder = vid2img(vid_name)