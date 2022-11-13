#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: 徐聪
# datetime: 2022-11-13 19:47
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