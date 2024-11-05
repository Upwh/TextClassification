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
                        max_iter=50,  # 最大迭代次数
                        activation='relu',  # 激活函数 默认为relu
                        solver='adam',  # 优化算法 默认为adam
                        batch_size='auto',  # 批处理大小 默认为'auto'
                        verbose=1,  # 显示信息 10次显示一次
                        learning_rate_init=0.001,  # 学习率 默认为0.001
                        early_stopping=True,  # 是否使用早停 默认为False
                        validation_fraction=0.1,  # 验证集比例 默认为0.1 用于计算模型性能判断是否早停
                        n_iter_no_change=10,  # 早停轮数 默认为10 次数内准确率不变则停止训练
                        tol=1e-4,  # 误差最小值 默认为1e-4
                        random_state=42,)
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
