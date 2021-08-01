# KD-tree
## 概述
KD Tree是密度聚类（DBSCAN）算法中计算样本和核心对象之间距离来获取最近邻以及KNN算法中用于计算最近邻的快速、便捷构建方式。
当样本数据量少的时候，我们可以使用brute这种暴力的方式进行求解最近邻，即计算到所有样本的距离。但是当样本量比较大的时候，直接计算所有样本的距离，工作量有点大，所以在这种情况下，我们可以使用KD Tree来快速的计算。

## 创建方式
KD树采用从m个样本的n维特征中，分别计算n个特征取值的方差，用方差最大的第k维特征作为根节点。
对于这个特征，选择取值的中位数作为样本的划分点，对于小于该值的样本划分到左子树，对于大于等于该值的样本划分到右子树，对左右子树采用同样的方式找方差最大的特征作为根节点，递归即可产生KD树。

## 实例
https://blog.csdn.net/gongxifacai_believe/article/details/104828520 参考自这篇博客，终于把KD-tree讲明白了
- 二维样本：{(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)}，由此可以画出一个二叉树。

- 当我们生成KD树以后，就可以去预测测试集里面的样本目标点了。对于一个目标点，我们首先在KD树里面找到包含目标点的叶子节点。以目标点为圆心，以目标点到叶子节点样本实例的距离为半径，得到一个超球体，最近邻的点一定在这个超球体内部。然后返回叶子节点的父节点，检查另一个子节点包含的超矩形体是否和超球体相交，如果相交就到这个子节点寻找是否有更加近的近邻,有的话就更新最近邻。如果不相交那就简单了，我们直接返回父节点的父节点，在另一个子树继续搜索最近邻。当回溯到根节点时，算法结束，此时保存的最近邻节点就是最终的最近邻。  
代码和注释如下所示：

```python

from sklearn.neighbors import KDTree
import numpy as np

X = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
tree = KDTree(X)
# 使用KD树把平面上的点划分成为不同的区域，每个区域只有一个点
print("tree=",tree)
print("X[0]=",X[0])
dist, ind = tree.query([X[0]], k=2)
# 查找最邻近的点的距离和对应点的索引值。
#第一个参数是待测点的坐标，第二个参数k表示要查询的最近点的个数。K=2代表查询最邻近的两个点的距离及其索引值
print(dist)
print(ind)
receiver_list = tree.query_radius(X,r=3.5)
# 凡是距离小于r的点都会被返回出来
print("receiver_list=",receiver_list)
```