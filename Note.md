## 二、Regression

### ML的三个步骤

1. Model：定义一个model即function set，通常从最简单的线性函数开始逐渐加级数
2. Goodness of Function：定义一个损失函数（Loss Function）去评估model的好坏
3. 选择Best Function（Gradient Descent）

#### Step1：模型假设

假设 **线性模型 Linear model**：$y=b+∑w_{i}x_{i}$

$x_{i}$：各种特征值，身高、体重、种类等

$w_{i}$：每个特征值的权重

$b$：偏移量

后续以单个特征$x_{cp}$的线性模型为例，即模型为：$y=b+w\cdot x_{cp}$



#### Step2：模型评估

定义损失函数：

$L(f)=\sum^{10}_{n=1}(\hat{y}^{n}-f(x^{n}_{cp}))^{2}$

等价于 $ L(w,b)=\sum^{10}_{n=1}(\hat{y}^{n}-(b+w\cdot x^{n}_{cp}))^{2} $

即：真实值-估计值

则当前目标为：选择一个在训练集上最优的model $f^{*}$如下：

$f^{*}=\arg \min _{f} L(f)$
等价于 $w^{*}, b^{*}=\arg \min _{w, b} L(w, b)= \arg \min _{w, b}(\hat{y}^{n}-(b+w\cdot x^{n}_{cp}))^{2}$

<u>**理解**：这个问题可以通过多元方程、最小二乘等方法求得参数值，但利用Gradient Descent方法可以处理所有可微分的$L()$</u>



#### Step3：最佳模型（Gradient Desent）

一、对于只有一个参数的$L(w)$，$w^{*}=\arg \min _{w} L(w)$

1. 随机选择一个初始值$w_{0}$
2. 计算$\left.\frac{d L}{d w}\right|_{w=w^{0}}$
3. $w^{1} \leftarrow w^{0}-\left.\eta \frac{d L}{d w}\right|_{w=w^{0}}$，  $\eta$：learning rate
4. 计算$\left.\frac{d L}{d w}\right|_{w=w^{1}}$
5. $w^{2} \leftarrow w^{1}-\left.\eta \frac{d L}{d w}\right|_{w=w^{·}}$，  $\eta$：learning rate
6. 持续迭代……

<u>**理解**：斜率为负→最优点在当前点右侧→增加$w$</u>

​			<u>斜率为正→最优点在当前点左侧→减少$</u>w$

​			<u>对于曲线可能到达局部最优解，但对于Linear Regression中损失函数是凸的，一定可以找到最优解</u>

<img src="pic/image-20211124205058693.png" alt="image-20211124205058693" style="zoom: 33%;" />

二、对于只有多参数的$L(w,b)$，$w^{*},b^{*}=\arg \min _{w,b} L(w,b)$

1. 随机选择一个初始值$w_{0},b_{0}$
2. 计算$\left.\frac{\partial L}{\partial w}\right|_{w=w^{0}, b=b^{0}}$，$\left.\frac{\partial L}{\partial b}\right|_{w=w^{0}, b=b^{0}}$
3. $w^{1} \leftarrow w^{0}-\left.\eta \frac{\partial L}{\partial w}\right|_{w=w^{0}, b=b^{0}}, b^{1} \leftarrow b^{0}-\left.\eta \frac{\partial L}{\partial b}\right|_{w=w^{0}, b=b^{0}}$  ,     $\eta$：learning rate
4. 计算$\left.\frac{\partial L}{\partial w}\right|_{w=w^{1}, b=b^{1}}$，$\left.\frac{\partial L}{\partial b}\right|_{w=w^{1}, b=b^{1}}$
5. $w^{2} \leftarrow w^{1}-\left.\eta \frac{\partial L}{\partial w}\right|_{w=w^{1}, b=b^{1}}, b^{2} \leftarrow b^{1}-\left.\eta \frac{\partial L}{\partial b}\right|_{w=w^{1}, b=b^{1}}$  ,      $\eta$：learning rate
6. 持续迭代……

此时gradient为：$\nabla L= \left[\begin{array}{l}\frac{\partial L}{\partial w} \\ \frac{\partial L}{\partial b}\end{array}\right]$



### 性能验证

计算testing data和模型输出之间的ave error

得到次数从1-5的五个model的error如下：

<img src="pic/image-20211124210638787.png" alt="image-20211124210638787" style="zoom: 67%;" />

**注**：由于高次数model可以表示低次数model，高次数model的function space包含低次数model的function space



则不同model在训练集和测试集上的error如下：

<img src="pic/image-20211124210904911.png" alt="image-20211124210904911" style="zoom:67%;" />

如上，高次数model在测试集上过拟合，可见需要增加数据（发掘更多特征）

### 步骤优化

#### Step1优化：先修改模型，考虑多种feature

$y=\sum^{4}_{i=1}(b_{i}+\delta(x_{s}=i)w_{i}x_{cp}) $

$x_{s}=i$：代表4个物种类型

注：如果考虑了物种特征之后还在测试集上效果不好，则增加特征（身高、体重……）

#### **Step2优化：若增加数据后扔过拟合，则Regularization：**

修改Loss Functionn如下：

$L=\sum_{n}(\hat{y}^{n}-(b+\sum w_{i}x_{i}))^{2}+\lambda\sum(w_{i})^{2} $

| **$\sum_{n}(\hat{y}^{n}-(b+\sum w_{i}x_{i}))^{2}$** | **error** |
| --------------------------------------------------- | --------- |
| $\lambda\sum(w_{i})^{2}$                            | smooth    |

$\lambda$为常数，自己设置

**对smooth的解释：**

当输入增加$\Delta x_{i}$时，输出增加$w_{i}\Delta x_{i}$，则$\lambda\sum(w_{i})^{2}$项越小，输出对于输入的变化越不敏感，function越平滑，干扰的影响越小



### 正则化结果

<img src="pic/image-20211124213302076.png" alt="image-20211124213302076" style="zoom:67%;" />

可以看出：

$\lambda$越大，**训练集**的损失函数越倾向于考虑smooth而不管error，则error越大

$\lambda$越大，**测试集**先抗干扰，error减小，后对于error不敏感，则error越大



### Bias and Variance

数据和best function输出之间的error包含：bias和variance

#### Bias和variance的估计

假设随机变量$x$的均值和方差为$(\mu,\sigma^{2})$，样本为$\{x^{1},x^{2},\dots x^{N} \}$

$m=\frac{1}{N} \sum_{n} x^{n}$，$ s^{2}=\frac{1}{N} \sum_{n}\left(x^{n}-m\right)^{2}$

则：

$E[m]=E\left[\frac{1}{N} \sum_{n} x^{n}\right]=\frac{1}{N} \sum_{n} E\left[x^{n}\right]=\mu$

$Var[m]=\frac{\sigma^{2}}{N}$

$E[s^{2}]=\frac{N-1}{N}\sigma^{2}\ne \sigma^{2} $→大多数情况下$s^{2}<\sigma^{2}$



#### Bias和variance的区别

<img src="pic/image-20211124214539513.png" alt="image-20211124214539513" style="zoom:67%;" />

$\hat{f}$：最优函数（实际的靶心）

$f^{*}$：根据训练集得到的函数，由训练数据决定，每次可能不一样（每次射击的点）

$\bar{f}$：$f^{*}$的期望（训练时瞄准的靶心）

则：

**bias**：$\bar{f}$和$\hat{f}$之间的差距

**variance**：$f^{*}$和$\bar{f}$之间的差距



### 同一环境采集不同测试集的best function

<img src="pic/image-20211124215408540.png" alt="image-20211124215408540" style="zoom:67%;" />



#### Variance分析

<img src="pic/image-20211124215452748.png" alt="image-20211124215452748" style="zoom:67%;" />

模型越简单，variance越小

模型越复杂，variance越大

**总结：简单的模型更不容易受样本数据影响**



#### Bias分析

<img src="pic/image-20211124215720256.png" alt="image-20211124215720256" style="zoom:67%;" />

blue：$\bar{f}$

black：$\hat{f}$

可以看出：

模型越复杂，$\bar{f}$越接近$\hat{f}$，这是由于简单模型的function set较小，其space可能未包含$\hat{f}$



#### Bias和Variance的影响

<img src="pic/image-20211124220016393.png" alt="image-20211124220016393" style="zoom:67%;" />

简单模型：Bias大（Underfitting，欠拟合），Variance小

复杂模型：Bias小，Variance大（Overfitting，过拟合）



#### 如何解决Bias大

<u>Bias大的定义：model无法拟合训练集</u>

解决方法：

- 重新设计模型：增加特征参数、增加次数



#### 如何解决Variance大

<u>Variance大的定义：model可以拟合训练集但测试集error大</u>

解决方法：

- 增加数据：采集、自己制造（改造现有数据、生成新的数据……）
- Regularization：平滑化，在损失函数后加smooth，自己设置$\lambda$大小调整平滑程度，但这样调整了function space，可能损害bias



### Model选择步骤

<img src="pic/image-20211124221322831.png" alt="image-20211124221322831" style="zoom:67%;" />

Cross Validation步骤：

1. 用training set去训练所有model
2. 用Validation set去验证所有model的error
3. 接着用整个Training set在best model上apply
4. 此时public上的测试集error可以反映private上的测试集error



N-fold Cross Validation步骤：![image-20211124221444645](pic/image-20211124221444645-16377669367522.png)



## 三、Gradient Descent

Gradient： Loss的等高线的法线方向

### Tip1：调节学习速率

#### 学习速率对Loss的影响

1. 速率（步长）太小会导致收敛速度很慢
2. 速率（步长）太大会导致无法收敛到最优点，甚至可能导致Loss增加

<img src="pic/image-20211126133659395.png" alt="image-20211126133659395" style="zoom:67%;" />

#### 自适应调节学习速率

1. 初始时用较大的学习速率
2. 几次更新后按一定规律减少学习速率，e.g. $\frac{1}{t}$ decay：   $\eta^{t}=\frac{\eta}{\sqrt{t+1}}$
3. 不同的参数需要设置不同的学习率



#### Adagrad

中心思想：用每个参数先前迭代的均方根来分割其learning rate

即：$w^{t+1}\gets  w^{t}-\frac{\eta^{t}}{\sigma^{t}}g^{t}$

$\sigma^{t}=\sqrt{\frac{1}{t+1}\sum_{i=0}^{t}(g^{t})^{2}}$：参数$w$先前迭代的均方根

$\eta^{t}=\frac{\eta}{\sqrt{t+1}}$   

$g^{t}=\frac{\partial L(\theta^{t})}{\partial w}$

<img src="pic/image-20211126135303395.png" alt="image-20211126135303395" style="zoom:67%;" />

代入化简得：$w^{t+1}\gets  w^{t}-\frac{\eta}{\sqrt{\sum_{i=0}^{t}(g^{t})^{2}}}g^{t}$

<img src="pic/image-20211126135812793.png" alt="image-20211126135812793" style="zoom:67%;" />

**矛盾点：**梯度越大时，按照分子计算其步长越大，但按照分母计算其步长越小

**直观原因：**步长取决于当前梯度在历史梯度迭代中的相对大小如下

<img src="pic/image-20211126150527184.png" alt="image-20211126150527184" style="zoom:67%;" />

**数学解释：**

<img src="pic/image-20211126151608565.png" alt="image-20211126151608565" style="zoom:67%;" />

只考虑一个参数时，$x_{0}$点的微分值与其距最低点的距离成正比

<img src="pic/image-20211126151752754.png" alt="image-20211126151752754" style="zoom:67%;" />

但多个方向的微分（梯度）不能一起比较

<img src="pic/image-20211126151931718.png" alt="image-20211126151931718" style="zoom:67%;" />

则最佳步长为：|一次微分|/二次微分

<img src="pic/image-20211126152018317.png" alt="image-20211126152018317" style="zoom:67%;" />

由于一次微分可以估计二次微分，即可以通过$\sqrt{\frac{1}{t+1}\sum_{i=0}^{t}(g^{t})^{2}}$估计二次微分的值，故Adgrad



### Tip2：随机梯度下降

之前的梯度下降：$L=\sum_{n}\left(\hat{y}^{n}-\left(b+\sum w_{i} x_{i}^{n}\right)\right)^{2}$

缺点：计算所有训练样本的Loss和

随机梯度下降：$L=\left(\hat{y}^{n}-\left(b+\sum w_{i} x_{i}^{n}\right)\right)^{2}$

优点：只计算一个样本的损失函数，参数更新快，但随机性高

<img src="pic/image-20211126153414229.png" alt="image-20211126153414229" style="zoom:67%;" />

### Tip3：特征缩放

对于一个两特征的function：$y=b+w_{1}x_{1}+w_{2}x_{2}$

如果$w_{1}$和$w_{2}$的scaling不一样，则变化相同的$\Delta w$对$y$的影响程度不一样

<img src="pic/image-20211126154111457.png" alt="image-20211126154111457" style="zoom:67%;" />

下面两个圆分别为scaling 前和scaling 后的error surface图

**scaling前：**不同方向必须使用不同的learning rate，采用adgrad等方法

**scaling后：**参数变化稳定，更新简单，路径更直接指向最低点，效率更高



#### 常见的scaling方法

<img src="pic/image-20211126154517945.png" alt="image-20211126154517945" style="zoom:67%;" />

对于有$R$个特征，每个特征为$n$维的function，对每一维的特征都求平均和均方根

然后将第$r$个特征的第$i$维进行如下归一化操作：

$x_{i}^{r}=\frac{x_{i}^{r}-m_{i}}{\sigma_{i} }$

此时：所有特征每个dimension的均值为0方差为1



### Gradient Descent Theory

Q：当使用梯度下降法时，Loss一定会随着参数更新而减小吗？

A：不一定，学习速率过大是Loss反而会增加

梯度下降的思想可以理解为：寻找每次更新参数为圆心的圆内Loss最小的一个点

问题：如何寻找一个圆内Loss最小的一点？

<img src="pic/image-20211126160915196-16379146957563.png" alt="image-20211126160915196" style="zoom:67%;" />

#### 数学解释

泰勒展开式定义

<img src="pic/image-20211126161305153.png" alt="image-20211126161305153" style="zoom:67%;" />

拓展到多变量泰勒展开为：

<img src="pic/image-20211126161638592.png" alt="image-20211126161638592" style="zoom:67%;" />

回到之前如何快速在圆圈内找到最小值。基于泰勒展开式，在 (a,b)(a,b)(a,b) 点的红色圆圈范围内，可以将损失函数用泰勒展开式进行简化：

<img src="pic/image-20211126161914438.png" alt="image-20211126161914438" style="zoom:67%;" />

将问题进而简化为下图：

<img src="pic/image-20211126161929658.png" alt="image-20211126161929658" style="zoom:67%;" />

不考虑s的话，可以看出剩下的部分就是两个向量$(\Deltaθ_{1},\Deltaθ_{2})$和 $(u,v)$ 的内积，那怎样让它最小，就是和向量$(u,v)$ 方向相反的向量

<img src="pic/image-20211126162047234.png" alt="image-20211126162047234" style="zoom:67%;" />

然后将$u$和$v$带入。

<img src="pic/image-20211126162107522.png" alt="image-20211126162107522" style="zoom:67%;" />

$L(\theta)\approx s+ u(\theta_{1} - a)+ v(\theta_{2} - b) $

发现最后的式子就是梯度下降的式子。但这里用这种方法找到这个式子有个前提，泰勒展开式给的损失函数的估算值是要足够精确的，而这需要红色的圈圈足够小（也就是学习率足够小）来保证。

这里只考虑了泰勒展开式的一次项，如果考虑到二次项（比如牛顿法），在实际中不是特别好，会涉及到二次微分等，多很多的运算，性价比不好。



### 梯度下降的限制

![img](https://datawhalechina.github.io/leeml-notes/chapter6/res/chapter6-23.png)

容易陷入局部极值 还有可能卡在不是极值，但微分值是0的地方 还有可能实际中只是当微分值小于某一个数值就停下来了，但这里只是比较平缓，并不是极值点





