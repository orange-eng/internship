# Deep Learning Notes

**构建机器学习算法**

都是简单的秘方： 数据集、代价函数、优化过程和模型


**Dropout**
- 虽然Dropout在特定模型上每一步的代价是微不足道的，但在一个完整的系统上使用Dropout的代价可能非常显著。因为Dropout是一个正则化技术，它减少了模型的有效容量。为了抵消这种影响，我们必须增大模型规模。不出意外的话，使用Dropout时最佳验证集的误差会低很多，但这是以更大的模型和更多训练算法的迭代次数为代价换来的。对于非常大的数据集，正则化带来的泛化误差减少得很小。在这些情况下，使用Dropout和更大模型的计算代价可能超过正则化带来的好处。
- 只有极少的训练样本可用时，Dropout不会很有效。在只有不到 5000 的样本的Alternative Splicing数据集上 (Xiong et al., 2011)，贝叶斯神经网络 (Neal, 1996比Dropout表现得更好
- Dropout说的简单一点就是：我们在前向传播的时候，让某个神经元的激活值以一定的概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征

**对抗样本与对抗训练**
- 对抗样本，就是会使得机器学习的算法产生误判的样本
- 对抗训练，通过在原有的模型训练过程中注入对抗样本，提升模型对微小扰动的鲁棒性

**病态**
- 病态问题一般被认为存在于神经网络训练过程中。病态体现在随机梯度下降会‘‘卡’’ 在某些情况，此时即使很小的更新步长也会增加代价函数  
链接： https://blog.csdn.net/foolsnowman/article/details/51614862

**局部最小值**
- 一种能够排除局部极小值是主要问题的检测方法是画出梯度范数随时间的变化。如果梯度范数没有缩小到一个微小的值，那么该问题既不是局部极小值，也不是其他形式的临界点。

**预训练模型**
- 训练模型来求解一个简化的问题，然后转移到最后的问题，有时也会更有效些。这些在直接训练目标模型求解目标问题之前，训练简单模型求解简化问题的方法统称为 预训练。

**卷积的效率**
- 两个图像的高度均为 280 个像素。输入图像的宽度为 320 个像素，而输出图像的宽度为 319个像素。这个变换可以通过包含两个元素的卷积核来描述，使用卷积需要 319 ×280 ×3 = 267, 960次浮点运算（每个输出像素需要两次乘法和一次加法）。
- 为了用矩阵乘法描述相同的变换，需要一个包含 320 × 280 × 319 × 280 个或者说超过 80 亿个元素的矩阵，这使得卷积对于表示这种变换更有效 40 亿倍。直接运行矩阵乘法的算法将执行超过 160 亿次浮点运算，这使得卷积在计算上大约有 60,000 倍的效率。

**收集更多的数据**
- 如果更大的模型和仔细调试的优化算法效果不佳，那么问题可能源自训练数据的质量。数据可能含太多噪声，或是可能不包含预测输出所需的正确输入。这意味着我们需要重新开始，收集更干净的数据或是收集特征更丰富的数据集。

**超参数**
- 学习率可能是最重要的超参数。如果你只有时间调整一个超参数，那就调整学习率。相比其他超参数，它以一种更复杂的方式控制模型的有效容量——当学习率适合优化问题时，模型的有效容量最高，此时学习率是正确的，既不是特别大也不是特别小。  

**表示学习**
- 表示学习特别有趣，因为它提供了进行无监督学习和半监督学习的一种方法。我们通常会有巨量的未标注训练数据和相对较少的标注训练数据。在非常有限的标 注数据集上监督学习通常会导致严重的过拟合。半监督学习通过进一步学习未标注数据，来解决过拟合的问题。具体地，我们可以从未标注数据上学习出很好的表示，然后用这些表示来解决监督学习问题。

- 在非常有限的标 注数据集上监督学习通常会导致严重的过拟合。半监督学习通过进一步学习未标注数据，来解决过拟合的问题

**蒙特卡洛算法**
- 蒙特卡罗方法又称统计模拟法、随机抽样技术，是一种随机模拟方法，以概率和统计理论方法为基础的一种计算方法，是使用随机数（或伪随机数）来解决很多计算问题的方法。将所求解的问题同一定的概率模型相联系，用电子计算机实现统计模拟或抽样，以获得问题的近似解。为象征性地表明这一方法的概率统计特征，故借用赌城蒙特卡罗命名。

- https://blog.csdn.net/narcissus2_/article/details/99647407

**卡尔曼滤波**
抽空学习了一下卡尔曼滤波，有一些不一样的体验。这里也简单记录一下。  
链接（国外）：http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/  
链接（国内）：https://blog.csdn.net/u010720661/article/details/63253509

**标准差的计算**
先放上代码，用于求标准差：  
```python
import numpy as np
arr = [1,2,3,4,5,6]
#求均值
arr_mean = np.mean(arr)
#求方差
arr_var = np.var(arr)
#求标准差
arr_std = np.std(arr,ddof=1)
```
在统计学中，  
- 如果是总体，标准差公式根号内除以 n；
- 如果是样本，标准差公式根号内除以（n-1）  

numpy 的 .std() 和 pandas 的 .std() 函数之间是不同的。
- numpy 计算的是总体(母体)标准差，参数ddof = 0。
- pandas 计算的是样本标准差，参数ddof = 1。

如果我们知道所有的分数，那么我们就有了总体——因此，要使用 pandas 进行归一化处理，我们需要将“ddof”设置为 0。
