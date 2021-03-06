# -*- coding: utf-8 -*-
"""ViratKohli_T20_ScorePredictor.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10BDCnJttqnX5OZHteGJRdFmtEFk6QtGT
"""

import pandas as pd
import numpy as np
import pickle

vk = pd.read_csv('ViratKohli_T20_ScorePredictor - Sheet1.csv')

x = np.array(vk.iloc[:, 0:1])
y = np.array(vk.iloc[:, 1:])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=21)



from sklearn.linear_model import LinearRegression
vk_lr=LinearRegression()

vk_lr.fit(x_train,y_train)




pickle.dump(vk_lr, open('score.pkl', 'wb'))
