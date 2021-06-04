import cv2 as cv
from PIL import Image
import pytesseract as tess
 
 
def recoginse_text(image):
    """
    步骤：
    1、灰度，二值化处理
    2、形态学操作去噪
    3、识别
    :param image:
    :return:
    """
 
    # 灰度 二值化
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    # 如果是白底黑字 建议 _INV
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV| cv.THRESH_OTSU)
 
 
    # 形态学操作 (根据需要设置参数（1，2）)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(1,2))  #去除横向细线
    morph1 = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 1)) #去除纵向细线
    morph2 = cv.morphologyEx(morph1,cv.MORPH_OPEN,kernel)
    cv.imshow("Morph",morph2)
 
    # 黑底白字取非，变为白底黑字（便于pytesseract 识别）
    cv.bitwise_not(morph2,morph2)
    textImage = Image.fromarray(morph2)
 
    # 图片转文字
    text=tess.image_to_string(textImage)
    print("识别结果：%s"%text)
 
 
def main():
 
    # 读取需要识别的数字字母图片，并显示读到的原图
    src = cv.imread("1.png")
    cv.imshow("src",src)
 
    # 识别
    recoginse_text(src)
 
    cv.waitKey(0)
    cv.destroyAllWindows()
 
if __name__=="__main__":
    main()
