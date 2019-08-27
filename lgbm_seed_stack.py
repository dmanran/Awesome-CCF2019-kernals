# @FROM:https://zhuanlan.zhihu.com/p/79687336
#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import catboost as cbt
from sklearn.metrics import accuracy_score, roc_auc_score,log_loss
import gc
import math
import time
from tqdm import tqdm
import datetime
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
import warnings
import os
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
pd.options.display.max_rows = None

train = pd.read_csv('../first_round_training_data.csv')
test = pd.read_csv('../first_round_testing_data.csv')
submit = pd.read_csv('../submit_example.csv')
dit = {'Excellent':0,'Good':1,'Pass':2,'Fail':3}
train['label'] = train['Quality_label'].map(dit)

featherlist = ["Parameter"+str(i) for i in range(1,11)]
print(featherlist)
X_train = train[featherlist]
y = train["label"]

X_test = test[featherlist]
# prepare the train data

# train the best model with 5kfold

# train the model with blending
oof = np.zeros((X_train.shape[0],4))# 线下验证
prediction = np.zeros((X_test.shape[0],4))# 线上结论
seeds = [2255,2266,223344, 2019 * 2 + 1024,332232111, 40,96, 20,48, 1,80247,8,5,3,254,54,3434,2424,23,222,22222,222223332,222,222,2,4,32322777,8888]
num_model_seed = 15
print('training')
for model_seed in range(num_model_seed):
    print('模型',model_seed + 1,'开始训练')
    oof_cat = np.zeros((X_train.shape[0],4))
    prediction_cat=np.zeros((X_test.shape[0],4))
    skf = StratifiedKFold(n_splits=5, random_state=seeds[model_seed], shuffle=True)
    for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
        print(index)
        train_x, test_x, train_y, test_y = X_train.iloc[train_index], X_train.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        train_data=lgb.Dataset(train_x,label=train_y)
        validation_data=lgb.Dataset(test_x,label=test_y)
        gc.collect()
        params = {
            'learning_rate':0.01,
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class':4,
            'metrix': 'multi_logloss',
            'learning_rate': 0.002296,
            'max_depth': 6,
            'num_leaves': 10,
            'feature_fraction': 0.4,
            'bagging_fraction': 0.6,
            'bagging_freq': 17
        }
        lgbmodel=lgb.train(params,train_data,valid_sets=[validation_data],num_boost_round = 10000,verbose_eval = 1000,
                           early_stopping_rounds = 1000)
#         cbt_model.fit(train_x, train_y ,eval_set=(test_x,test_y))
        oof_cat[test_index] += lgbmodel.predict(test_x)
        prediction_cat += lgbmodel.predict(X_test)/5
        gc.collect()
    oof += oof_cat / num_model_seed
    prediction += prediction_cat / num_model_seed
    print('logloss',log_loss(pd.get_dummies(y).values, oof_cat))
    print('ac',accuracy_score(y, np.argmax(oof_cat,axis=1)))
print('logloss',log_loss(pd.get_dummies(y).values, oof))
print('ac',accuracy_score(y, np.argmax(oof,axis=1)))
sub = test[['Group']]
prob_cols = [i for i in submit.columns if i not in ['Group']]
print(prob_cols)
for i, f in enumerate(prob_cols):
    sub[f] = prediction[:, i]
for i in prob_cols:
    sub[i] = sub.groupby('Group')[i].transform('mean')
sub = sub.drop_duplicates()
sub.to_csv("submission_mean.csv",index=False)
