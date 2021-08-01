# internship

## 百度面试经验

**百度CCL面试**  
- 面试官很和气，一开始就直奔主题，甚至没有让我自我介绍，直接开始问问题了。面试官是一个研究机器人的，对物理模型和仿真比较了解。整个面试持续了40min,神清气爽，很开心，而且还学习到了很多东西。没有涉及到leetcode编程问题。（赶紧多练）  

### 1.先来讲讲你在本科阶段的研究经历吧
这一部分之前准备好了稿子，所以都比较顺畅。面试的关注点都集中在机器人部分。我说我负责过云台电机的双闭环PID算法，于是就开始了这一部分的解释说明。其中问道一个问题说：为什么速度环就只需要PI即可，而角度环只需要P？我没有答上来。按理说应该是，速度环更加追求准确性和灵敏性，所以积分项十分重要，另外速度环的运动往往是一阶的，比比例环要复杂。
### 2.是否可以使用神经网络来学习物理模型？给我的感觉就是学习如何调参的问题，但是也不是单纯的超参数。
比如一个球抛出，那么如何去在模拟器上计算这个球的运动轨迹？这个其实比较简单，但是如果推广到1000个球，那么应该如何计算呢？这些都是问题。如果我并不关注整个模型的内部过程，更关注结果，是否可以使用学习的方式来拟合输出呢？这个显得有些深奥，我只能往损失函数的方向上扯了。说了半天也没有说到点，但是面试官一直很亲和，而且更像是在跟我探讨一些问题。
### 3.如果使用模拟器来计算1000个球的数据就很慢，是否可以使用学习的方法来加速这个过程呢？
其实这一部分并不知道想要回答什么东西，但是也要硬着头皮和面试官聊天，坚持住就好。不要让气氛显得尴尬就行。我理解的仿真主要是涉及到初始条件，约束方程和参数设置。那么如何使用这些东西来完成学习？确实不清楚
### 4.最后还问了一下，为什么要使用C和C++混编？
因为C++可以实现面向对象的编程方式，便于封装成库。问我是否使用过C++的内置库boost和bullet，这个确实没有过。

面试官比较在意面试的时间，不满足3个月，时间还是太短了。所以，感觉竞争力不够大。


## 商汤面试经验

### AutoML面试
**面试官非常专业**  
一开始先是简单的自我介绍，然后面试官发了一道编程的题目，主要是关于反卷积网络transpose conv的简单编程。但是只可以使用矩阵运算的方式，不可使用深度学习框架。我在这里磨了30min都没做出来，实在是太尴尬了。由此可见，对底层代码的掌握有多么重要。另外在线上做题的时候，还不可以切换到其他界面，否则会被记录到，也比较尴尬。

之后就开始问问简历上的问题了。
- 1.简单讲讲你的实习经历和项目经历吧  
此处就可以疯狂介绍自己的项目经历。
- 2.针对超分辨率，那么超分辨率常用的几种损失函数，你在设计网络的过程中是如何选择的呢？  
- 3.你在复现这些论文的时候是怎样的思路？  
- 4.针对目标检测，你是如何处理负样本的？负样本和正样本指的是真实框与预选框的重叠程度，设置一个阈值，如果超过这个阈值则该预选框被认为是一个负样本
- 5.对一个抽烟检测， 其实是一个二分类任务，没必要使用YOLOv5这种网络吧？（这个问题我真的没有思考过）

让我头疼的问题都在上面了。最后还询问了一下他们做NAS的思路，尤其是关于超参数搜索的问题，还是可以学到很多东西。
整个面试持续了一个小时，让我很尴尬，发现自己的编程能力实在是太差了，每天虽然在做很多框架性质的东西，但是事实上从来没有关注过底层代码的实现，被彻底打脸了。

### 基础视觉组面试
**很专业，被打击到了**
- 先自我介绍1min左右
- 开始就目标检测开始提问。
  - 你简单说说目标检测的训练过程
  - yolov4,yolov3中的哪些改进让你印象深刻？
  - 目标检测当中的label是如何处理的
  - 你对当前目标检测的发展趋势如何看待？
  - 非极大抑制的原理？
  - 在处理数据时，CPU会占用大多数时间，而GPU其实是在等CPU，这种状态如何处理呢？
  - pytorch中的worker参数的意义  
这里补充一下num_workers的作用
```python
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True, num_worker=4)
valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs, num_worker=4)
```
dataloader一次性创建num_worker个worker，（也可以说dataloader一次性创建num_worker个工作进程，worker也是普通的工作进程），并用batch_sampler将指定batch分配给指定worker，worker将它负责的batch加载进RAM。然后，dataloader从RAM中找本轮迭代要用的batch，如果找到了，就使用。如果没找到，就要num_worker个worker继续加载batch到内存，直到dataloader在RAM中找到目标batch。一般情况下都是能找到的，因为batch_sampler指定batch时当然优先指定本轮要用的batch。

  - 多进程部署任务，是否有接触过？
  - Adam优化器和随机梯度下降的差别是什么？
  - 能够缓解过拟合的方法有哪些？
答案：https://blog.csdn.net/weixin_43455338/article/details/104885402
  - 随机梯度下降是如何工作的？

- 也问了一点关于超分辨率的知识
  - 你对超分辨率现在发展的状态做一个评价
  - 使用的损失函数有哪些？
  - 数据集是如何获得的？


#### **编程题**  

**走迷宫问题**
0代表可以走，1代表障碍，2代表终点，编写程序完成走迷宫

**非极大抑制**
编程非极大抑制的函数
- 答案：https://blog.csdn.net/weixin_44791964/article/details/106222846


#### 整体感觉

自己认识的太肤浅了，很多东西可能自己只有一个模糊的印象，但是当面试官让你现场写代码的时候，也会一头雾水，十分尴尬。经过本次面试，我总结了自己的不足有如下几点：
- 有点自以为是，总觉得自己什么都懂，其实都很肤浅，以这种状态显然是进不了商汤的
- 经典算法的一些原理都不是很懂，导致现场写不出广度优先搜索的内容
- 目标检测中的细节要加强，一旦遇到同行，就会被一些很细节的问题所困扰。这里最好是整理一下计算机视觉当中最经典的几种模块，然后不断复习。说实话，自己真的好菜哦
- 每次都被代码打败，好伤心，还是要多练，Leetcode走起来走起来


## 地平线面试经验

**地平线实习**

就直接是电话交流，了解了一下基本情况，因为我只能实习2个月左右，所以时间周期比较短，这总是一个弊端，之前没考虑过。
然后发了一个题目，是复现一篇文章的内容，先试试吧。


### 任务
复现论文《Distribution Adaptive INT8 Quantization for Training CNNs》https://arxiv.org/abs/2102.04782  
时间
从你看到这份文档起的一周内。  
一些可能有用的提示  
- 1、推荐使用mmdet（https://github.com/open-mmlab/mmdetection）或detectron2（https://github.com/facebookresearch/detectron2）
- 2、可以利用一切你能找到的资源。
- 3、遇到任何问题，欢迎随时联系面试官寻求解决办法。当然如果你能自己解决，会更好。
- 最后说一句
任务的难度不小，如果你做出来了，说明你在未来的实习工作中，也可以表现得很优秀。期待一周后，你惊艳的report。
我们在地平线等你。