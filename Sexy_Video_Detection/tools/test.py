


import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D

# 简单方法画出漂亮的圆柱体（半径和高度均为1）

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
 
# 生成圆柱数据，底面半径为r，高度为h。
# 先根据极坐标方式生成数据
u = np.linspace(0,2*np.pi,50)  # 把圆分按角度为50等分
h = np.linspace(0,5,20)        # 把高度1均分为20份
x = np.outer(np.sin(u),np.ones(len(h)))  # x值重复20次
y = np.outer(np.cos(u),np.ones(len(h)))  # y值重复20次
z = np.outer(np.ones(len(u)),h)   # x，y 对应的高度

# Plot the surface
ax.plot_surface(x, y, z, cmap=plt.get_cmap('rainbow'))
 
plt.show()
