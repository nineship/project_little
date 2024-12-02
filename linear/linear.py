# 从 sklearn.datasets 导入波士顿房价数据读取器。
#from sklearn.datasets import load_boston # 数据集
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection as modsel
from sklearn import linear_model
import matplotlib.pyplot as plt
 
# 从读取房价数据存储在变量 boston 中
#boston = load_boston()
import sklearn
from sklearn import datasets #导入数据集合
 
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
#使用 Pandas 读取位于指定URL的CSV文件。它跳过了前22行（其中包含元数据）并读取剩余的数据。它还指定值之间的分隔符为一个或多个空白字符（\s+），并且文件中没有标题。
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
#创建特征矩阵 data。它选择每隔一个行（从第一行开始）并将其与每隔一个行（从第二行开始）连接起来，有效地将相邻的行合并成一对。然后丢弃第二组行的最后一列。这个操作本质上是将两行合并成一行，形成特征矩阵。
target = raw_df.values[1::2, 2]
#从每隔一个行（从第二行开始）中提取目标变量，并从每个这些行中选择第三列（索引为2）。这创建了目标数组。
 
# 输出数据描述。
# print(data_url.DESCR)
#该数据共有 506 条记录，13 个特征，没有缺失值
print(data.shape)
print(target.shape)
#划分训练集测试集
# 从sklearn.model_selection 导入数据分割器。
from sklearn.model_selection import train_test_split
X = data
y = target
# 随机采样 25% 的数据构建测试样本，其余作为训练样本。
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.25)
# 分析回归目标值的差异。
print("最高房价：", np.max(target))
print("最低房价：",np.min(target))
print("平均房价：", np.mean(target))
linreg = linear_model.LinearRegression()
linreg.fit(X_train, y_train)
# linreg.predict(x_train) 是预测值
print('预测值的方差：\t' + str(metrics.mean_squared_error(y_train, linreg.predict(X_train))))
print('预测值的确定系数（R^2）：\t' + str(linreg.score(X_train, y_train)))
#R平方值意义是趋势线拟合程度的指标，它的数值大小可以反映趋势线的估计值与对应的实际数据之间的拟合程度，R平方值是取值范围在0～1之间的数值，当趋势线的 R 平方值等于 1 或接近 1 时，其可靠性最高，反之则可靠性较低。
 
y_pred = linreg.predict(X_test)
rateofpred=metrics.mean_squared_error(y_test,y_pred)
print("均方误差：",rateofpred)
#通常,mean_squared_error越小越好。
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(10,6))
plt.plot(y_test, linewidth=3, label='ground truth', color = "blue", linestyle="-")
plt.plot(y_pred,linewidth=3,color = "red",label='predicted')
plt.legend(loc='best')
plt.xlabel('test data point')
plt.ylabel('target value')
plt.show()