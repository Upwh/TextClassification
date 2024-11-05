# 项目文件介绍

> README.md：项目介绍<br>
> images：README.md所使用图像<br>
> 20news-bydate_py3.pkz：sklearn下载并解析后的数据集<br>
> Classification.py：文本分类任务的主程序<br>
> Regression.py：曲线拟合任务的主程序<br>
> training_loss_42_27.png：文本分类任务训练种子42，训练次数27的训练过程中的损失函数值变化图

# 实验一：基于神经网络的新闻文本分类

## 一、实验目的

* 熟悉文本分类的基本原理；
* 熟悉神经网络训练过程及相关参数；
* 学会评估神经网络模型的性能；
* 了解相关库的使用方法。

## 二、实验原理

### 1. 分类问题

分类和回归任务是机器学习常见的任务类型，通常情况下分类问题输出的值是离散的，回归问题输出的值是连续的。 分类任务旨在预测离散的标签或类别，常用的方法叫逻辑回归，其将输入特征的线性组合结果输入到sigmoid函数中，输出一个介于0和1之间的概率值，代表分类的置信度。

在损失函数的使用中，分类和回归相差较大。本次实验使用的是交叉熵损失函数，对于单个数据点：

$$
H(p, q) = -\sum_{x} p(x) \log q(x)
$$

其中，$ p $ 是真实标签的概率分布，$ q $ 是模型预测的概率分布。在分类问题中，$ p $ 通常是一个 one-hot 编码的分布，即正确类别的概率是 1，其他类别的概率是 0；而 $ q $ 是模型预测每个类别的概率。

对于整个数据集，交叉熵损失是所有数据点损失的平均值：

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{M} y_{i,c} \log(\hat{y}_{i,c})
$$

其中，$ N $ 是样本数量，$ M $ 是类别数量，$ y_{i,c} $ 是第 $ i $ 个样本第 $ c $ 类的真实标签（0 或 1），$ \hat{y}_{i,c} $ 是模型预测第 $ i $ 个样本第 $ c $ 类的概率。

> 对于该公式的理解：损失函数需要是放大错误，加强反向传播过程中梯度的更新效率。对于该损失函数，对于负样本0的 $H $ 值为0；而对于正样本1，若其预测错误即预测值 $\hat{y}_{i,c}$ 较小时，其 $H$ 值应该较大，所以使用 $\log(\hat{y}_{i,c})$ 来让其与预测概率成反比来提高损失函数。

文本分类显然是一个分类任务，本实验使用Sklearn库中的MLPClassifier类来实现多层感知机分类器，最终的输出有20个通道，每个通道对应一个主题，输出概率值越大，则该主题的置信度越高。

### 2. 数据集介绍

fetch_20newsgroups 是一个在机器学习库 scikit-learn 中用于加载 20 个新闻组数据集的函数。这个数据集包含了大约 18000 篇新闻组文章，涵盖了20 个不同的主题，因此被称为 20 newsgroups 数据集 ([fetch_20newsgroups官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html))。<br>

> 通过`from sklearn.datasets import fetch_20newsgroups`可以获取数据集。调用数据集中的属性`target_names`可以观察到20个主题名称，`data`属性获取原文信息。

### 3. 数据预处理

原文为字符串形式，需要进行预处理，将其转换为数字形式，以便于神经网络模型的训练。sklearn库中采用TF-IDF（Term Frequency-Inverse Document Frequency）算法进行文本预处理。

> TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度，其中TF是词频(Term Frequency)，IDF是逆文本频率指数(Inverse Document Frequency)。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。TF-IDF加权的各种形式常被搜索引擎应用，作为文件与用户查询之间相关程度的度量或评级。    ——《百度百科》

$TF-IDF(t,d)=TF(t,d) * IDF(t)=\frac{n_{t,d}}{N_d} * log\frac{N}{n_t}$

其中 $n_{t,d}$ 表示词t在d文档出现次数， $N_d$ 表示文档d中所有词的出现次数总和， $N$ 是语料库中文档总数， $n_t$ 表示包含词t的文档数。

> 通过该公式可以将该数据集中所有词统计得到一个数字向量。在`sklearn`库中使用`TfidfVectorizer().fit_transform()`函数即可完成转换。通过观察得到的数据矩阵`(18846, 173762)`可以判断其包含了18000篇文章的173762个词的TF-IDF值。

### 4. 多层感知机

多层感知机（MLP，Multilayer Perceptron）是神经网络的一种类型，由至少一个隐藏层组成，每一层都有多个神经元，每个神经元都与上一层的所有神经元相连，并通过激活函数进行非线性变换，最后输出预测结果，详细内容参考课程PPT。<br>
本次实验采用最简单的网络结构，即只有一层隐藏层。根据以上对数据集的介绍，可以得到网络结构如下：

![MLP](./images/MLP.png)

## 三、实验步骤

### 1. 新建项目

打开Pycharm，选择新建项目，设置Conda环境，Python版本3.10。

![NewPrj](./images/CreateNewPrj.png)

### 2. 安装必需库

#### 1）pip 库配置

pip是一个现代的，通用的Python管理工具。提供了对 Python 包的查找、下载、安装、卸载的功能，接下来的必需库均使用pip进行安装。由于大部分必需库均来自国外服务器，所以需要更改pip的下载源，配置方法如下。

通过`ALT+F12`或Pycharm左下角图标![Terminal](./images/Terminal.png)打开Pycharm中内嵌的终端，如下图。

![OpenedTerminal](./images/TerminalOpened.png)

随后在终端输入以下内容：`pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple` 完成换源，终端会输出其将设置内容写入了指定文件中。

#### 2）必需库的安装

输入以下内容：

```
pip install scikit-learn
pip install matplotlib
```

安装过程可能较为缓慢，等待完成即可。

#### 3）数据集的下载安装

sklearn库调用fetch_20newsgroups函数会自动下载，但会出现因网络原因无法下载的情况，所以以下介绍通过需要手动下载并通过sklearn解析安装。

本项目中包含了`20news-bydate_py3.pkz`为已解析后的数据集，直接将其放入`C:\Users\用户名\.scikit_learn_data`目录即可，如下图所示。

![Datasets](./images/Datasets.png)

> scikit_learn_data目录在调用fetch_20newsgroups函数时会自动创建，未使用前需手动创建。

### 3. 程序创建及运行

在Pycharm项目目录下创建Classification.py文件，输入以下程序([MLPClassifier官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html))：

```python
# 数据集
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# 神经网络模型
from sklearn.neural_network import MLPClassifier
# 模型评估
from sklearn.metrics import accuracy_score
# 保存模型
import joblib
# 数据绘制
import matplotlib.pyplot as plt
# 系统工具
import os


# 加载所有数据集
newsgroups = fetch_20newsgroups(subset='all')
# 提取特征(文本信息转换为向量特征)
vectorizer = TfidfVectorizer()  # TF-IDF特征提取器
X = vectorizer.fit_transform(newsgroups.data)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, newsgroups.target, test_size=0.25, random_state=42)

# 加载已有模型
if os.path.exists('mlp_42_27.pkl'):
    mlp = joblib.load('mlp_42_27.pkl')
else:  # 创建分类神经网络模型
    mlp = MLPClassifier(hidden_layer_sizes=(100,),  # 隐藏层神经元数目 1层 100个
                        max_iter=100,  # 最大迭代次数
                        activation='relu',  # 激活函数 默认为relu
                        solver='adam',  # 优化算法 默认为adam
                        batch_size='auto',  # 批处理大小 默认为'auto'
                        verbose=1,  # 显示信息 10次显示一次
                        learning_rate_init=0.001,  # 学习率 默认为0.001
                        early_stopping=True,  # 是否使用早停 默认为False
                        validation_fraction=0.1,  # 验证集比例 默认为0.1 用于计算模型性能判断是否早停
                        n_iter_no_change=10,  # 早停轮数 默认为10 次数内准确率不变则停止训练
                        tol=1e-4,  # 误差最小值 默认为1e-4
                        random_state=42)
    # 训练模型
    mlp.fit(X_train, y_train)

# 数据绘制
losses = mlp.loss_curve_
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('training_loss_42_27.png')
plt.show()
# 保存模型
joblib.dump(mlp, 'mlp_42_27.pkl')
# 预测测试集
y_pred = mlp.predict(X_test)
# 打印对比预测结果和实际结果
print("Predicted labels:\n", y_pred)
print("True labels:\n", y_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

```

即可运行程序。

通过本项目包含的main.py可以直接运行程序。

> 除此之外，main.ipynb可以在jupyter环境下逐步运行程序，便于观察运行结果，且jupyter的变量将存储在工作空间，可以直接通过Pycharm查看。（jupyter仅能在Pycharm Professional版本运行`:(`，若想使用可自行安装或通过Anaconda自带jupyter配置环境）

程序的首次运行会创建单隐藏层100神经元的神经网络模型，并训练100次（27次后停止训练），保存模型到mlp_42_27.pkl文件中，并绘制训练过程中的损失函数值变化图。该模型已训练好，由于体积较大未上传github，保存至[百度网盘](https://pan.baidu.com/s/1ce6nrPBK865Tnj6LaFi93Q?pwd=qqry)。
运行结果如图`training_loss_42_27.png`。

![loss](training_loss_42_27.png)

### 4. 模型参数调整

通过调整`MLPClassifier`中模型的参数可以得到更好的模型效果，通过自行训练观察训练过程，分析各参数的影响。

### 四、心得体会👇

# 实验二：基于神经网络的曲线拟合

## 一、实验目的

* 了解曲线拟合的基本原理；
* 了解曲线拟合的相关方法；
* 了解回归问题的基本原理；

## 二、实验原理

曲线拟合（Curve Fitting）是指根据已知数据点，利用某种函数（如曲线、直线、抛物线等）来逼近这些数据点。曲线拟合的目的是找到一个函数，使得函数与已知数据点之间的误差（差距）最小。

### 回归问题

回归问题（Regression）是指预测一个连续变量（如实数）的输出值，即根据输入变量预测输出变量。回归问题的目标是找到一个函数或模型，使得输出变量与输入变量之间的差距最小。

在本实验中，我们将使用线性回归（Linear Regression）来拟合数据。线性回归是一种简单而有效的回归方法，其假设输入变量与输出变量之间存在线性关系。

与分类问题不同，本次回归问题使用的损失函数是均方误差（Mean Squared Error，MSE）。MSE是输入变量与输出变量之间的差距的平方的平均值，其公式如下：

$$
MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

其中，$N$ 是样本数量，$y_i$ 是第 $i$ 个样本的真实输出值，$\hat{y}_i$ 是模型预测的输出值。显然，该公式表示的是两组数据点的差距，差距越大损失越大，代表两组数据点拟合程度越差。

## 三、实验步骤

### 1. 程序实现

无需新建项目，使用实验一环境即可。[MLPRegressor官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)

创建Regression.py文件，输入以下程序运行观察结果：

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neural_network import MLPRegressor
from joblib import dump, load
import time
# 设置中文字体
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
# 设置正常显示符号
matplotlib.rcParams['axes.unicode_minus'] = False


# 创建回归训练数据 sin曲线
x = np.arange(0, 10, 0.01).reshape(-1, 1)  # 输入数据 可调整数据量
y = np.sin(x).ravel()  # 真实输出数据 1维数组

# 绘制原始数据
plt.plot(x, y, 'o')
plt.show()

# 加载模型
# model = load('MLPRegressor.joblib')

# 使用sklearn创建多层感知机回归模型
model = MLPRegressor(hidden_layer_sizes=(100, 100, ),  # 隐藏层神经元数量 2层 每个隐藏层100神经元 尝试其他隐藏层结构
                     warm_start=True,  # 启用warm_start 接续训练
                     activation='relu',  # 激活函数 默认relu
                     random_state=42,  # 随机种子
                     max_iter=50,  # 最大迭代次数 50次
                     learning_rate='constant')  # 学习率调整策略 默认constant不变

# 训练模型 500次迭代 每50次输出一次损失函数值 显示训练效果
for i in range(10):
    # 经测试 由于多次分段训练 模型会在损失较低时出现振荡
    # 因此这里只训练10次 作为拟合过程的展示
    model.fit(x, y)  # 训练模型
    print(f'第{50*(i+1)}次迭代 损失函数值：{model.loss_}')

    # 绘制曲线
    y_pred = model.predict(x)
    plt.plot(x, y, 'o')
    plt.plot(x, y_pred, '-')
    plt.title(f'第{50*(i+1)}次迭代 损失函数值：{model.loss_}')
    plt.show()
    time.sleep(1)  # 延时1s 防止过快的绘制请求报错

    # 保存模型
    dump(model, 'MLPRegressor.joblib')
plt.plot(model.loss_curve_)
plt.title('500次迭代 损失函数图')
plt.show()

# 作为对比 一次性训练500次
model = MLPRegressor(hidden_layer_sizes=(100, 100, ),  # 隐藏层神经元数量 2层 每个隐藏层100神经元 尝试其他隐藏层结构
                     activation='relu',  # 激活函数 默认relu
                     random_state=42,  # 随机种子
                     max_iter=500,  # 最大迭代次数 50次
                     learning_rate='adaptive')  # 学习率调整策略 默认constant不变
model.fit(x, y)  # 训练模型
y_pred = model.predict(x)
plt.plot(x, y, 'o')
plt.plot(x, y_pred, '-')
plt.title(f'一次性训练500次 损失函数值：{model.loss_}')
plt.show()
# 损失函数图
plt.plot(model.loss_curve_)
plt.title('一次性500次迭代 损失函数图')
plt.show()
```

### 2. 模型参数调整

通过调整`MLPRegressor`中模型的参数可以得到更好的模型效果，通过自行训练观察训练过程，分析各参数的影响。

## 四、心得体会👇
