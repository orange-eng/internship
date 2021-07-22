import cv2 as cv

img = cv.imread('./examples/orange1.png')

#img.resize((256,256))
img = cv.resize(img,(256,256))

cv.imwrite("./examples/orange.jpg",img)