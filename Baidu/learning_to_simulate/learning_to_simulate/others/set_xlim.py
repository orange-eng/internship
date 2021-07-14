

import matplotlib.pyplot as plt
import numpy as np



fig, ax = plt.subplots()
x=np.arange(10)
y=x
ax.plot(x,y)
#给x轴,y轴设置最小值，最大值的变量范围
bounds = [[0.1, 0.9], [0.1, 0.9]]
ax.set_xlim(bounds[0][0], bounds[0][1])
ax.set_ylim(bounds[1][0], bounds[1][1])

# ax.set_xlim(0,5)
# ax.set_ylim(0,6)
plt.show()
