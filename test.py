#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: 徐聪
# datetime: 2022-11-07 0:11
# software: PyCharm
import sys

import matplotlib.pyplot as plt
import numpy as np

# x = range(1, 27)
# y = [1, 2, 4, 8, 16, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 21, 22, 23, 24, 25, 26, 1, 2, 4, 8]
import pandas as pd

x = range(0, 19)
y = np.array([1,2,3,4,2,3,4,5,6,3,4,2,3,4,2,3,4,5,2])
y2 = np.array([1,2,3,4,2,3,4,5,6,3,4,2,3,4,2,3,4,5,2])

df = pd.DataFrame(np.array([y, y2]).T).sort_values(0).values
print(df)
# cost = np.dot(y-np.average(y), (y-np.average(y)).T)
# print(cost)

# plt.figure()
# plt.plot(x, y, 'ro-')
# plt.grid()
# plt.xlabel("RTT")
# plt.ylabel("cwnd")
# plt.show()
# print(sum(y[:7]))
