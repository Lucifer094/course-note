[toc]

# 参考资料：

B站视频：[(强推)李宏毅2021春机器学习课程](https://www.bilibili.com/video/BV1Wv411h7kN?p=1)

GitHub笔记：[DeepLearning_LHY21_Notes](https://github.com/unclestrong/DeepLearning_LHY21_Notes)

# 1.机器学习基本概念(P2)

## 1.1.机器学习分类

回归任务（regression）

分类任务（classfication）

结构化学习（structured learning）

## 1.2.实际问题机器学习过程

解决预测视频点击量预测

1.训练：training data

1）定义模型：线性函数（linear model）

2）定义损失函数：MAE(mean absolute error)，MSE(mean square error)

3）最优化(optimization)：梯度下降(gradient descent)

一个参数时的梯度下降法
$$
w^1=w^0-\eta \frac{\partial L}{\partial w}|_{w=w^0}\\
$$

$\eta$ ：学习率（Learning rate）
$\eta$ ：超参数（hyperparameters）

可能遇到的问题：Local minima，Global minima

多个参数时梯度下降法
$$
w^1=w^0-\eta \frac{\partial L}{\partial w}|_{w=w^0}\\
b^1=b^0-\eta \frac{\partial L}{\partial b}|_{b=b^0}\\
$$


2.测试：test data

# 2.深度学习基本概念(P3)

## 2.1.Model Bias

Model Bias的产生为仅用简单的线性函数无法描述真实情况。

All Piecewise Linear Curves = constant + sum of a set of (Linear Model)

## 2.2.Sigmoid Function
$$
y = c \frac{1}{1+e^{-(b+wx_1)}} \\
=c*sigmoid(b+wx_1)
$$

![](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220102153500.png)

## 2.3.Hard Sigmoid

![image](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220102153501.png )

## 2.4.New Model:More Features

$$
y = b + w x_1 \\
y = b + \sum_i c_i * sigmoid(b_i + w_i x_1)
$$

$$
y = b + \sum_j w_j x_j \\
y = b + \sum_i c_i * sigmoid(b_i + \sum_j w_{i,j} + x_j)
$$

![image](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220102153502.png)

## 2.5.Deep learning(Neural Network)

为什么选择深度而不选择宽度增加？

Overfitting：过拟合

设定了很多种类型的模型后，经过训练后，如何选择合适的模型。

# 3.使用Pytorch训练模型整体过程(P4)

## 3.1.起始设定：

```python
dataset = MyDataset(file)	# read data via MyDataset
tr_set = Dataloader(dataset, 16, shuffle=True)	# put dataset into Dataloader
model = MyModewl().to(device)	# contruct model and move to device
criterion = nn.MSELoss()	# set loss function
optimizer = torch.optim.SGD(model.parameters(), 0.1)	# set optimizer
```

## 3.2.开始训练：

```python
for epoch in range(n_epochs):	# iterate n_epochs
    model.train()				# set model to train mode
    for x, y in tr_set:			# iterate through the dataloader
        optimizer.zero_grad()	# set gradient to zero
        x, y = x.to(device), y.to(device)	# move data to device(cpu/cuda)
        pred = model(x)			# forward pass (compute output)
        loss = criterion(pred, y)	# compute loss
        loss.backward()			# compute gradient (backpropagation)
        optimizer.step()		# update model with optimizer
```

每次训练完一个epoch后，使用validation set（注：training set的shuffle=True，表示每一次epoch的数据顺序不一致，采用随机的方式，validation set的shuffle=False，表示Dataloader的提取数据方式为固定顺序，防止计算evaluation的结果因为数据顺序的改变而不一致）进行一次evaluation：

```python
model.eval()						# set model to evaluation mode
total_loss = 0
for x, y in dv_set:					# iterate through the dataloader
    x, y = x.to(device), y.to(device)	# move data to device (cpu/cuda)
    with torch.no_grad():			# disable gradient calculation
        pred = model(x)				# forward pass (compute output)
        loss = ctiterion(pred y)	# compute loss
    total_loss += loss.cpu().item()*len(x)	# accumulate loss
    avg_loss = total_loss / len(dv_set.dataset)	# compute averaged loss
```

在模型经过训练和验证后，挑选一个最为合适的模型，根据测试数据进行预测（该示例为预测模型，测试数据集中没有标准答案，仅需要将测试数据输入，并把预测结果保存下来即可）：

```python
model.eval()						# set mofel to evaluation model
preds = []
for x in tt_set:					# iterater through the dataloader
    x = x.to(device)				# move data to device (cpu/cuda)
    with torch.no_grad():			# disable gradient calculation
        pred = model(x)				# forward pass (compute output)
        preds.append(pred.cpu())	# collect prediction
```

## 3.3.保存模型：

```python
torch.save(model.state_dict(), path)
```

## 3.4加载模型：

```python
ckpt = torch.load(path)
model.load_state_dict(ckpt)
```

# 4.反向传播(P9)

## 4.1.反向传播基础原理(chain rule)

Case 1:
$$
y=g(x) \\
z=h(y) \\
\frac{dz}{dx} = \frac{dz}{dy} \frac{dy}{dx}
$$


Case 2:
$$
x=g(s) \\
y=h(s) \\
z=k(x,y) \\
\frac{dz}{ds} = \frac{\partial z}{\partial x} \frac{dx}{ds} + 
\frac{\partial z}{\partial y} \frac{dy}{ds}
$$

# 5.网络调整(P10)

## 5.1.整体思路

训练完一次模型后，根据**训练集的损失**以及**测试集的损失**来确定网络需要进行**模型改进**还是**优化函数改进**。

整体调整方向如下图所示：

![image](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220102153503.png)

## 5.2.Model Bias

出现Model Bias的情况为模型在**训练集上的误差较大**，如图所示：

![image](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220102153503.png)

原因：模型太过简单。

解决方式：

1.**重新设计一个model,给model更大的弹性**，增加输入的features。

2.用Deep Learning,来增加model的弹性。

## 5.3.Optimization Issue

出现**训练集上的误差较大**的原因除了Model Bias以外，还可能是**Optimization**的问题。

出现的原因是在利用**Gradient descent**寻找最优解时，陷入局部最优解，如图所示：

![image](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220102153504.png)

如何判断是**Optimization**的问题：

![image](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220102153505.png)

如图所示：**56层网络**的训练集上的损失函数要高于**20层网络**的损失函数，表明不是模型不够复杂，而是**Optimization**的问题。

建议：

1.**看到一个你从来没有做过的问题,也许你可以先跑一些比较小的,比较浅的network,或甚至用一些,不是deep learning的方法**。

2.**If deeper networks do not obtain smaller loss on training data, then there is optimization issue.**

举例子：

![image](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220102153506.png)

如图所示，网络在1->4层的逐层增加过程中，损失不断降低，然而在5层时损失函数反而增加，表示使用的**Optimization**有问题。

## 5.4.Overfitting

表现形式：**training的loss小,testing的loss大**。

极端情况进行解释原因：训练出的模型只是记住了训练集中的输入和输出的对应关系，对于未知的部分输出是一个随机的，这种情况下在训练集中的损失为0，测试集中的损失极高。

解决方式：

1.**增加训练集**：可以使用**data augmentation**的方式进行数据扩充，例如图像中的翻转，放大等操作（上下颠倒的操作很有可能会造成数据在真实世界中的无意义，所以很少用）。

2.**限制模型的弹性**：

1)**给较少的参数**：在fully-connected network中，减少神经元的个数。（**注**：**CNN是一个比较有限制的架构**；**fully-connected network**的限制较少，可以通过减少参数的形式来增加模型的限制）

2)使用**较少的features**：

3)**Early stopping**：训练设置提前结束的条件。（另还有**Regularization**正则化的方式）

4)**Dropout**：将部分神经元进行禁用。

## 5.5.Cross Validation

将训练集区分为**Training Set**和**Validation Set**两部分，用**Validation Set**的分数去挑选模型。

如何将数据集进行划分并挑选出合适的模型？

**N-fold Cross Validation**：

1.将数据集划分为N个部分，其中包括M组数据，每组数据中都包含P个部分的验证集以及$\frac{N}{M}-P$个部分的测试集。

2.用每组数据对模型进行训练和测试，将分数进行平均，挑选出最合适的模型。

3.之后再将所有的数据当作训练集进行训练，得到最终的模型。

如下图所示过程：

![image](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220102153507.png)

## 5.6.Mismatch

表现：和Overfitting表现相似，训练集表示好，测试集表现差。

成因：由于训练集数据的分布和测试集数据的分布不一样导致。

训练集和测试集数据如下图所示：

![image](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220102153508.png)

# 6.梯度为0时该如何判断当前的情况(P11)

## 6.1.梯度为0时可能的情况

1)局部最小值local minima（训练维度大，出现可能极少）

2)最大值

3)鞍点saddle point（可解决）

4)全局最小值（**目标**）

![](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220102153509.png)

## 6.2.利用hessian matrix判定当前点的情况

泰勒展开式Tayler Series Approximation：

![](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220102153510.png)

当梯度$g$为0时，式子简化：

![](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220102153511.png)

使用Hessian Matrix来进行判断当前是local minima，local maxima还是saddle point。

![](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220102153512.png)

根据线性代数知识，可以转化为讨论Hessian Matrix的特征值（eigen value）的正负情况。

# 7.梯度为0时该怎么办(P12)

## 7.1.使用Batch

### 7.1.1.不同大小的Batch size网络更新情况

以训练集中包含20个数据下，极端情况为例：

1.当Batch为1时，相当于每一个Epoch中要模型进行20次的优化，网络参数更新时间短、频率快，但是每次训练所得到的梯度方向不一定是朝着全局最优的方向走。

2.当Batch为20时，相当于每一个Epoch中模型只进行一次优化，网络更新参数时间长、频率慢，但是优化方向总体而言会朝向全局最优。

**注：**从更新参数的效果上看，大的Batch size会有利于参数更新的正确性。

### 7.1.2.并行运算后Batch size的大小和时间的使用情况

1.使用GPU并行加速运算后，一个大的Batch size在一定范围内，其更新一次网络参数所使用的时间，和Batch size为1的情况下所使用的时间较为接近。

2.对于一个Epoch来说，在一定范围内Batch size越大，所需时间越短，超过这个范围后，所需时间较为接近。

**注**：从运行速度上看，使用GPU并行后，更新一次网络参数时，大的Batch size在一定范围内也可以和小的Batch size使用相近的时间，并且大的Batch size可以更快的运行完一个Epoch。

### 7.1.3.小Batch size的优势

1.小Batch size由于每个数据的梯度方向不确定性，可以更有利于训练时越过局部最优值，达到全局最优。

2.小Batch size也更有利于模型的泛化(generalization)，大的Batch size所得到的局部最优解可能没有小Batch size所得到的局部最优解好。

### 7.1.4.大小Batch size的对比总结

![](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220102153513.png)

对于Batch size而言，大和小都有各自的优势和劣势，所以**Batch size的大小也就成了一个需要调整的Hyperparameter**。

##  7.2.Momentum

利用之前的梯度下降方向来使得越过当前局部最优值。

![](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220102153514.png)

可以理解为利用前一次的梯度下降方向去更新现在的梯度下降方向。

![](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220102153515.png)

也可以认为是每次进行梯度下降时，要考虑在此之前所有的梯度下降方向

![](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220102153516.png)

# 8.自适应学习率调整(Adaptive Learning Rate)

## 8.1.Traing stuck不等于Small Gradient

当训练停止时，不一定当时的梯度已经最小，反而可能会出现长时间训练后梯度来回震荡，有可能当时处于最低点的边缘，由于每一次的梯度下降跨步太大，导致无法到达最低点。

![](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220103160123.png)

## 8.2.采用固定的学习率$\eta$ 将会很难达到最优解

及时不面对critical point问题，当采用固定的学习率$\eta$ 时，也将很难训练到最优解。

1）采用较大的学习率$\eta$ 时：会在最优解附近来回横跳。

2）采用较小的学习率$\eta$ 时：会在开始时向最优解较为稳定的移动，但是在靠近最优解后，会移动的十分缓慢。

![](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220103162524.png)

## 8.3.不同的参数需要不同的$\eta$ 学习率

1）对于一个参数而言，使用常规的梯度下降法，参数更新方式为：
$$
\theta _{i}^{t+1} \longleftarrow \theta _{i}^{t} - \eta g _{i}^{t}
\\ g_{i}^{t} = \frac {\partial L} {\partial \theta _{i}} \mid _{\theta = \theta ^{t}}
$$


其中$\theta _{i}^{t} $表示$t$时刻参数$\theta _{i}$的数值，$\eta$表示学习率，$g _{i}^{t}$表示$t$时刻的梯度，$L$表示损失函数。

2）对于一个参数而言，使用自适应的学习率$\eta$ 的梯度下降法，参数更新的方式为：
$$
\theta _{i}^{t+1} \longleftarrow \theta _{i}^{t} - \frac {\eta}{\sigma _i ^t} g _{i}^{t}
$$


其中参数$\sigma _i ^t$是一个与参数$\theta _{i}$和时间$t$相关的参数。

## 8.4.learning rate的$\frac {\eta}{\sigma _i ^t}$中$\sigma _i ^t$的变式

### 8.4.1.Root Mean Square

整体思路：将历史梯度绝对值的大小进行考虑，使得随着网络的训练，learning rate的值越来越小，从而保证梯度在不断的训练过程中越来越小，达到最终收敛的目的，防止在最优解附近来回横跳。

![image-20210319150808494](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220112005731.png)

Root Mean Square的思路方式使用在Adagrad中。

![image-20210319160639783](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220112010304.png)

### 8.4.2.RMS Prop

整体思路：Root Mean Square方式中，learning rate随着训练不断降低，可以达到梯度随着训练逐步降低，最终达到收敛的效果。然而当一个参数在训练过程中，其梯度的大小往往会出现实大实小的情况，这个时候需要在梯度较小时采用较大的learning rate，在梯度较大时采用较小的learning rate，从而保证正确而快速的收敛。

![image-20210319212301760](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220112010800.png)

$\alpha$：表示当前梯度大小对于learning rate的影响大小，是一个超参数（hyperparameter）

### 8.4.3.Adam

整体思路：RMS Prop + Momentum

[论文链接](https://arxiv.org/pdf/1412.6980.pdf)

![image-20210319220458633](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220112012354.png)

## 8.5.learning rate的$\frac {\eta}{\sigma _i ^t}$中$\eta$的变式（又称为learning rate的scheduling）

出现的问题：如果单纯的将learning rate的$\frac {\eta}{\sigma _i ^t}$中的$\sigma _i ^t$部分进行使用动态调整，每次都将之前的梯度结果进行综合考虑。就会导致当梯度值的变化若是开始大、后来小时，当小梯度积累到一定程度时，由于$\sigma _i ^t$是一个指数为-1的幂函数，下降速度极快，其结果很快就会小于$\eta$值，而$\frac {\eta}{\sigma _i ^t}$也是一个指数为-1的幂函数，导致整体的learning rate极速增加，从而发生抖动，但随着新的大梯度的加入，会使得逐渐learning rate降低，最终梯度逐渐平稳。

![image-20210319221217246](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220112204803.png)

整体思路：只调整learning rate原始$\frac {\eta}{\sigma _i ^t}$的分母部分$\sigma _i ^t$已经不满足了，将分子部分$\eta$也进行调整，将其升级为与时间相关的一个变量$\eta ^t$。

![image-20210319222132512](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220112211441.png)

$\eta ^t$一般使用Warm Up的方式，随着时间先变大后变小。

![image-20210319223155277](https://gitee.com/zhangxin094/course-note/raw/master/HongYiLi2021/img/20220112212000.png)







