


import os
import cv2 as cv

img_name = os.listdir('img/check_in_dataset/JPEGImages')
label_name = os.listdir('img/check_in_dataset/Annotations')

# for i in range(len(img_name)):
#     print(img_name[i])

label_number = []
for i in range(len(label_name)):
    (filename, extension) = os.path.splitext(label_name[i])
    #print(filename)
    label_number.append(filename)
print(label_number)

for i in range(len(img_name)):
    (filename, extension) = os.path.splitext(img_name[i])
    if filename in label_number:
        _img = cv.imread('img/check_in_dataset/JPEGImages/{}'.format(img_name[i]))
        cv.imwrite('img/check_in_dataset/JPEGImages_filter/{}'.format(img_name[i]),_img)
        
