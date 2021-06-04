import os
class ImageRename():
    def __init__(self):
        self.path = r"img\\GOOD_CASE"  # 需要改名的文件

        self.path1 = r"img\\GOOD_CASE" # 改名后文件存在的路径

    # 'norain-1000x2.png', 'norain-1001x2.png',
    def re_name(self):
        filelist = os.listdir(self.path)
        #print(filelist)

        total_num = len(filelist)
        print("total_num=",total_num)

        i = 0
        for item in filelist:

            src = os.path.join(os.path.abspath(self.path), item)
            dst = os.path.join(os.path.abspath(self.path1), str(i) + '.jpg')
            os.rename(src, dst)
            print('converting %s to %s ...' % (src, dst))
            i = i + 1

        # for item in filelist:

		# 	# 对图片的名字进行分割和提取 
        #     number1= item.split(".")[0]
        #     first = number1.split("_")[0]
        #     second = number1.split("_")[1]

        #     first1 = int(first)
        #     second2 = int(second)
        # #     # number2 = int(number2)
        #     if item.endswith('.jpg'):

        #         if(second2==1):

        #             src = os.path.join(os.path.abspath(self.path), item)
        #             dst = os.path.join(os.path.abspath(self.path1), str(first1-900) + '.jpg')
        #             os.rename(src, dst)
        #             print('converting %s to %s ...' % (src, dst))

if __name__ == '__main__':
    newname = ImageRename()
    newname.re_name()
