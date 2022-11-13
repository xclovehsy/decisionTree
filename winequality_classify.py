#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: 徐聪
# datetime: 2022-11-04 13:04
# software: PyCharm

import pandas as pd
import numpy as np
import DecisionTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import plottree
import pickle

data_path = r"Dataset/winequality_data.xlsx"
tree_path = r"wine_dt.txt"
df = pd.read_excel(data_path)

X = pd.DataFrame(df).values[:, :-1]
y = pd.DataFrame(df).values[:, -1]
feature_name = list(df.columns)[:-1]
id2name = {}
name2id = {}
for i in range(len(feature_name)):
    id2name[i] = feature_name[i]
    name2id[feature_name[i]] = i

# # 归一化处理

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# 训练集和测试集切分
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 训练模型
dt = DecisionTree.DT()
dtree = dt.createDecTree(x_train, y_train, id2name)

# 模型准确性评估
print("剪枝前决策树准确率")
print(classification_report(y_test, dt.predict(dtree, x_test)))
#
# # 绘制决策树图像
# plottree.createPlot(dtree)

# 后剪枝
dt.postPruning(dtree, dtree, x_test, y_test)

# 模型准确性评估
print("剪枝后决策树准确率")
print(classification_report(y_test, dt.predict(dtree, x_test)))

# 绘制决策树图像
plottree.createPlot(dtree)

# 存储决策树
with open(tree_path, 'wb') as fw:
    pickle.dump(dtree, fw)

# dtree = {}
# # 读取决策树
# with open(tree_path, 'rb') as fr:
#     dtree = pickle.load(fr)
#
# # 模型准确性评估
# print("红酒数据集-决策树准确率")
# print(classification_report(y_test, dt.predict(dtree, x_test)))
#
# # 绘制决策树图像
# plottree.createPlot(dtree)
