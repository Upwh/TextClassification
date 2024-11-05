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
