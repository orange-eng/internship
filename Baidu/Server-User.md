# Sever Users

## 安装PaddleCloud客户端
- 先登录relay，然后再ssh
- 然后就可以安装Python,就直接copy语句即可。注意，最后加上--user

# 服务器使用步骤

1、登录relay

2、登录yp

把代码上传到服务器的方案：  
1.登录到远程服务器  
2.cd /home/public/share 进入到share文件夹中  
3.在浏览器打开网址http:
- 这里其实就是服务器上的share文件夹，两者是同步的
4.把本地代码（好像必须要打包之后才能上传）上传到浏览器的share文件夹  
5.在服务器上即可查看到该文件  
6.使用mv指令完成对代码的移动操作  

**tensorflow上检测GPU是否能用**

```python
import tensorflow as tf
 print('GPU',tf.test.is_gpu_available())
```

### 常见命令
- 安装常见指令
```python
tar -zxvf cudnn.tgz			#这里的文件名要对应上下载的cudnn文件名  
cd cuda 					# 此处进入cudnn解压的目录
#
cp ./include/cudnn.h ~/cuda-10.1/include		#复制粘贴  
cp ./lib64/libcudnn* ~/cuda-10.1/lib64			#复制粘贴
chmod a+r ~/cuda-10.1/include/cudnn.h ~/cuda-10.1/lib64/libcudnn*
#
vim ~/.bashrc     进入~/.bashrc  
摁住ctrl+g，直接跳到最后一行，摁一下i键，进入插入模式，现在可以编辑文档了。
source ~/.bashrc     更新bashrc使其生效
#
cat /etc/issue 查看centos版本	6.3
```
- 命令行常见指令
```python
nvcc -V               # 查看cuda版本
nvidia-smi            # 查看GPU版本
ctrl + l              # 清屏，类似clear命令
ls                    # 列出目录内容的意思
cat /proc/cpuinfo     # 显示CPU info的信息
date                  # 显示系统日期
cd ..                 # 返回上一级目录
mkdir dir1            # 创建一个叫做 ‘dir1’ 的目录’
rm -rf dir1           # 删除一个叫做 ‘dir1’ 的目录并同时删除其内容
mv dir1 new_dir       # 重命名/移动 一个目录
cp -a dir1 dir2       # 复制一个目录
zip file1.zip file1   #创建一个zip格式的压缩包
unzip file1.zip       #解压一个zip格式压缩包

ctrl + c              # 终止当前运行的程序

```


### 常见命令（docker）
- 查看容器数目
```python
docker images                       # 查看镜像
docker ps                           # 列出容器相关信息
docker exec -it caochengzhi bash    # 进入容器
exit                                # 退出容器
docker cp 源数据目录 目标目录        # 把源目录copy到目标目录
docker cp /www/runoob 96f7f14e99ab:/www/
#把主机/www/runoob目录拷贝到容器96f7f14e99ab的/www目录下。
docker cp  96f7f14e99ab:/www /tmp/
# 把容器96f7f14e99ab的/www目录拷贝到主机的/tmp目录中。
```

# 非root用户在服务器上安装cuda10.0+cudnn7.6.5+tensorflow-gpu=1.15+python3.7
总的来说，坑是非常多的。这里推荐几个博客，说的很到位。
- https://blog.csdn.net/dlh_sycamore/article/details/107600717
- https://blog.csdn.net/weixin_43689163/article/details/106555955
- https://blog.csdn.net/qq_35498453/article/details/110532839
- https://blog.csdn.net/m0_37548423/article/details/81173678
主要有下面几个步骤：
### 安装cuda10.0
- 直接去官网，选择合适的配置。我的选择是：
linux x86_64 Ubuntu  16.04  runfile
直接下载之后，上传到服务器即可

### 安装cudnn7.6.5
- 直接去官网选好 cuDNN for CUDA10.0
- 获得之后，解压，得到cuda文件，将其中的内容copy到CUDA的目录中
```python
cp ./include/cudnn.h ~/cuda-10.1/include		#复制粘贴
cp ./lib64/libcudnn* ~/cuda-10.1/lib64			#复制粘贴
chmod a+r ~/cuda-10.1/include/cudnn.h ~/cuda-10.1/lib64/libcudnn*
```
### 安装anaconda
- 安装anaconda3版本，主要是为了迎合python3.7   

**安装完anaconda之后，不能用conda指令：**  
输入 export PATH="~/anaconda3/bin:$PATH”修改环境变量即可  

**安装好之后，不能conda create**  
这是由于没有激活环境。输入代码：“source activate”即可进入虚拟环境

### 创建虚拟环境python=3.7
- 安装tensorflow-gpu==1.5

注意：  
**1.如果有两个CUDA，10.0和10.2,应该如何切换？**  
答：修改环境变量即可
```python
cp /home/dailh/cuda/include/cudnn.h  /home/dailh/cuda-10.0/include/  
cp /home/dailh/cuda/lib64/libcudnn*  /home/dailh/cuda-10.0/lib64
chmod a+r /home/dailh/cuda-10.0/include/cudnn.h  /home/dailh/cuda-10.0/lib64/libcudnn*
```
**2.出现“Could not load dynamic library "libcudart.so.10.0"”**  
答：这里指的是需要用CUDA10.0，不能用CUDA10.2,而且必须是python=3.7(亲测有效)，3.6是不行的

**3.各种版本之间的关系是怎样的**  
答：主要是CUDA版本和python版本要用对。python2.7是不能安装tensorflow1.15~2.0的。必须使用python3.7才行。安装CUDA10.0之后一定要修改好环境变量，并且安装对应的CUDNN。
