#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.optimizers import Adam
from pandas import concat
from pandas import read_csv
from helper import series_to_supervised, stage_series_to_supervised

from sklearn.svm import SVR


# In[4]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import PyPluMA

# ### Dataset

# In[6]:
class SVRPlugin:
 def input(self, inputfile):
  self.dataset = pd.read_csv(inputfile, index_col=0)

 def run(self):
     pass

 def output(self, outputfile):
  self.dataset.fillna(0, inplace=True)
  data = self.dataset[:]
  n_hours = 24*7
  K = 24
  stages = self.dataset[['WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']]
  stages_supervised = series_to_supervised(stages, n_hours, K)
  non_stages = data[['WS_S4', 'FLOW_S25A', 'FLOW_S25B', 'FLOW_S26', 'PUMP_S26', 'PUMP_S25B', 'MEAN_RAIN']]
  non_stages_supervised = series_to_supervised(non_stages, n_hours-1, 1)
  non_stages_supervised_cut = non_stages_supervised.iloc[24:, :]
  n_features = stages.shape[1] + non_stages.shape[1]   # 1 rainfall + FGate_S25A + FGate_S25B + FGate_S26 + 8WS + PUMP_S26
  non_stages_supervised_cut.reset_index(drop=True, inplace=True)
  stages_supervised.reset_index(drop=True, inplace=True)

  all_data = concat([
                   non_stages_supervised_cut.iloc[:, :],
                   stages_supervised.iloc[:, :]],
                   axis=1)
  all_data = all_data.values
  n_train_hours = int(len(all_data)*0.8)
  train = all_data[:n_train_hours, :]    # 0 column is the rainfall to measure heavy/medium/light
  test = all_data[n_train_hours:, :]
  n_obs = n_hours * n_features
  train_X, train_y = train[:, :n_obs], train[:, -stages.shape[1]*K:]
  test_X, test_y = test[:, :n_obs], test[:, -stages.shape[1]*K:]
  scaler = MinMaxScaler(feature_range=(0, 1))
  train_X = scaler.fit_transform(train_X)
  train_y = scaler.fit_transform(train_y)
  test_X = scaler.fit_transform(test_X)
  test_y = scaler.fit_transform(test_y)
  from sklearn.svm import SVR
  from sklearn.multioutput import MultiOutputRegressor
  import time
  svm_model = SVR(kernel='rbf', C=100, gamma=0.1, max_iter=5)
  multi_output_model = MultiOutputRegressor(svm_model)
  multi_output_model.fit(train_X, train_y)
  yhat = multi_output_model.predict(test_X)
  inv_yhat = scaler.inverse_transform(yhat)
  inv_y = scaler.inverse_transform(test_y)

  inv_yhat = pd.DataFrame(inv_yhat)
  inv_y = pd.DataFrame(inv_y)
  mae = mean_absolute_error(inv_yhat, inv_y)
  rmse = np.sqrt(mean_squared_error(inv_yhat, inv_y))
  inv_yhat.to_csv(outputfile+"/inv_yhat_svm.csv")
  inv_y.to_csv(outputfile+"/inv_y_svm.csv")
