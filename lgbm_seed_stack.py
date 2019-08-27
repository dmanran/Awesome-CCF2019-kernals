#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
import gc
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")
pd.options.display.max_columns = None
pd.options.display.max_rows = None

########################################################################################################################
# ORI: 处理类别标签并分割x,y_train
# d_train = pd.read_csv("../data/first_round_training_data.csv")
# d_test = pd.read_csv("../data/first_round_testing_data.csv")
# submit = pd.read_csv("../data/submit.csv")
#
# dit = {"Excellent": 0, "Good": 1, "Pass": 2, "Fail": 3}
# d_train["label"] = d_train["Quality_label"].map(dit)
#
# featherlist = ["Parameter" + str(i) for i in range(1, 11)]
# print(featherlist)
# x_train = d_train[featherlist]
# y_train = d_train["label"]
#
# x_test = d_test[featherlist]
d_train = pd.read_csv("../data/first_round_training_data.csv")
d_test = pd.read_csv("../data/first_round_testing_data.csv")
submit = pd.read_csv("../data/submit.csv")

x_train = d_train[["Parameter" + str(i) for i in range(1, 11)]]
x_test = d_test[["Parameter" + str(i) for i in range(1, 11)]]
y_train = d_train["Quality_label"].map(
    {"Excellent": 0, "Good": 1, "Pass": 2, "Fail": 3}
)
########################################################################################################################

# Note: f_ 表示当前fold的相关临时变量; s_ 表示当前seed的相关临时变量

y_valid = np.zeros((x_train.shape[0], 4))
y_pred = np.zeros((x_test.shape[0], 4))
seeds = np.random.randint(0, 2 ** 32, 15)
print("training")
for i, seed in enumerate(seeds):
    print("#" * 79)
    s_y_valid = np.zeros((x_train.shape[0], 4))
    s_y_test = np.zeros((x_test.shape[0], 4))
    skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

    for index, (f_index_train, f_index_valid) in enumerate(skf.split(x_train, y_train)):
        print("Model:{} Fold: {}".format(i + 1, index))
        f_x_train, f_x_valid, f_y_train, f_y_valid = (
            x_train.iloc[f_index_train],
            x_train.iloc[f_index_valid],
            y_train.iloc[f_index_train],
            y_train.iloc[f_index_valid],
        )
        f_train = lgb.Dataset(f_x_train, label=f_y_train)
        f_valid = lgb.Dataset(f_x_valid, label=f_y_valid)
        gc.collect()
        params = {
            # "learning_rate": 0.01,
            "boosting_type": "gbdt",
            "objective": "multiclass",
            "num_class": 4,
            "metric": "multf_logloss",
            "learning_rate": 0.002296,
            "max_depth": 6,
            "num_leaves": 10,
            "feature_fraction": 0.4,
            "bagging_fraction": 0.6,
            "bagging_freq": 17,
            "num_threads": 16,
        }
        f_evals_result = {}
        f_model = lgb.train(
            params,
            f_train,
            valid_sets=[f_valid],
            num_boost_round=10000,
            verbose_eval=1000,
            early_stopping_rounds=1000,
            evals_result=f_evals_result,
        )
        lgb.plot_metric(f_evals_result)

        # 收集当前fold的模型在vaild和test上的结果
        s_y_valid[f_index_valid] += f_model.predict(f_x_valid)
        s_y_test += f_model.predict(x_test) / 5
        gc.collect()

    # 收集当前seed的模型在vaild和test上的结果
    y_valid += s_y_valid / len(seeds)
    y_pred += s_y_test / len(seeds)
    print("logloss", log_loss(pd.get_dummies(y_train).values, s_y_valid))
    print("ac", accuracy_score(y_train, np.argmax(s_y_valid, axis=1)))

# 收集全部模型在vaild和test上的结果
print("logloss", log_loss(pd.get_dummies(y_train).values, y_valid))
print("ac", accuracy_score(y_train, np.argmax(y_valid, axis=1)))

########################################################################################################################
# ORI: 将结果丢进分组里
#
# sub = d_test[["Group"]]
# prob_cols = [i for i in submit.columns if i not in ["Group"]]
# print(prob_cols)
# for i, f in enumerate(prob_cols):
#     sub[f] = y_pred[:, i]
# for i in prob_cols:
#     sub[i] = sub.groupby("Group")[i].transform("mean")
# sub = sub.drop_duplicates()
y_pred = pd.DataFrame(
    y_pred, columns=["Excellent ratio", "Good ratio", "Pass ratio", "Fail ratio"]
)

y_pred["Group"] = d_test["Group"]
group_pred = y_pred.groupby("Group").mean()
# 作者并没有概率缩放
# for c in ["Excellent ratio", "Good ratio", "Pass ratio", "Fail ratio"]:
#     group_pred[c] /= group_pred.sum(axis=1)

group_pred.to_csv("./submission_mean.csv", index=False)
########################################################################################################################
