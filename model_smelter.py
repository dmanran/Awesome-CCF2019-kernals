# python3
# -*- coding: utf-8 -*-
# @File    : model_smelter.py
# @Desc    : 万能一把梭
# @Project : CCF-离散制造过程中典型工件的质量符合率预测
# @Time    : 8/27/19 1:47 PM
# @Author  : Loopy
# @Contact : peter@mail.loopy.tech
# @License : CC BY-NC-SA 4.0 (subject to project license)

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm.sklearn import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error as LOSS
from math import *
import warnings

warnings.filterwarnings("ignore")
sns.set(style="white", color_codes=True)

target = ["Excellent", "Good", "Pass", "Fail"]
seeds = np.random.randint(0, 2 ** 32 - 1, 3)
path = {
    "train": "../data/first_round_training_data.csv",
    "test": "../data/first_round_testing_data.csv",
    "submit": "../data/submit.csv",
    "model": "../temp_model",
    "res": "./res",
}


def run_model(model, model_type):
    global scale, log
    loss, i = [], 0
    for f_index_train, f_index_test in kfold.split(x_train):
        # split & scale(optional) the dataset
        if scale:
            m_x_train, m_y_train = (
                scaler.transform(x_train.iloc[f_index_train]),
                y_train.iloc[f_index_train],
            )
            m_x_test, m_y_test = (
                scaler.transform(x_train.iloc[f_index_test]),
                y_train.iloc[f_index_test],
            )
        else:
            m_x_train, m_y_train = (
                x_train.iloc[f_index_train],
                y_train.iloc[f_index_train],
            )
            m_x_test, m_y_test = (
                x_train.iloc[f_index_test],
                y_train.iloc[f_index_test],
            )

        # fit & evaluate the model
        model.fit(m_x_train, m_y_train)
        loss.append(LOSS(m_y_test, model.predict(m_x_test)))

        # print & save the model
        # print(i,loss[-1])

        joblib.dump(filename="../temp_model/" + str(model_type) + str(i), value=model)
        i += 1

    print(model_type, "Loss=", np.mean(loss))
    log.loc[model_type] = [np.cov(loss), np.mean(loss)]
    return {
        "cov": round(float(np.cov(loss)), 6),
        "mean": round(float(np.mean(loss)), 6),
        "model": model,
    }


d_test = pd.read_csv(path["test"], encoding="gbk")
d_train = pd.read_csv(path["train"], encoding="gbk")
# d_train = pd.read_csv("../data_wash/train.csv")
# d_test = pd.read_csv("../data_wash/test.csv")

# deal with Quality_label
for q in target:
    d_train[q] = d_train["Quality_label"].apply(lambda x: 1 if x == q else 0)
d_train = d_train.drop(["Quality_label"], axis=1)

# remove attribute
d_train = d_train.drop(["Attribute" + str(i) for i in range(1, 11)], axis=1)
group = d_test["Group"]
d_test = d_test.drop(["Group"], axis=1)

with open(path["res"] + "/seed_log.csv", "w") as f:
    f.write("seed,loss\n")

#########################################################################################
# GO!
for seed_i, seed in enumerate(seeds):
    score = []
    s_res = "{}/submit_{}.csv".format(path["res"], seed)
    print("=" * 79)

    # init a scaler
    scaler = StandardScaler().fit(d_train)
    scale = False

    # init a kfold
    splits = 15
    kfold = KFold(n_splits=splits, shuffle=True, random_state=seed)

    for catg in target:
        y_train = d_train[catg]
        x_train = d_train.drop(target, axis=1)
        # x_train = d_train.drop([catg], axis=1)
        print("! Working on seed{}/{} = {} category = {}".format(seed_i + 1, np.shape(seeds)[0], seed, catg))
        print("=" * 50)
        #####################################################################################################
        # train
        log = pd.DataFrame([], [], columns=["LOSS_Cov", "LOSS_Mean"])

        model = XGBRegressor()
        run_model(model, "XGB")

        model = GradientBoostingRegressor()
        run_model(model, "GBDT")

        model = LGBMRegressor()
        run_model(model, "LGBM")

        model = CatBoostRegressor(logging_level="Silent")
        run_model(model, "CAT")

        model = Lasso(alpha=0.0005)
        run_model(model, "LASSO")

        model = ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3)
        run_model(model, "ELS")

        print("=" * 50)
        print(log.sort_values("LOSS_Mean"))

        chose_model = "XGB GBDT LGBM CAT".split(" ")  # LASSO ELS

        loss_list = []
        for i in range(len(chose_model) + 1):
            loss_list.append([])

        i = 0
        for train, test in kfold.split(x_train):
            # split & scale(optional) the dataset
            if scale:
                f_x_train, f_y_train = (
                    scaler.transform(x_train.iloc[train]),
                    y_train.iloc[train],
                )
                f_x_test, f_y_test = (
                    scaler.transform(x_train.iloc[test]),
                    y_train.iloc[test],
                )
            else:
                f_x_train, f_y_train = x_train.iloc[train], y_train.iloc[train]
                f_x_test, f_y_test = x_train.iloc[test], y_train.iloc[test]

            # load the models and stack
            isFirst = True
            for model_name in chose_model:
                model = joblib.load(filename="../temp_model/{}{}".format(model_name, i))
                if isFirst:
                    pred = model.predict(f_x_test)
                    isFirst = False
                else:
                    pred = np.c_[pred, model.predict(f_x_test)]

            # fit lf model
            lr = Lasso(alpha=0.0005)
            lr.fit(pred, f_y_test)

            # record all the loss
            for j in range(len(chose_model)):
                loss_list[j].append(LOSS(pred[:, j: j + 1].flatten(), f_y_test))
            loss_list[j + 1].append(LOSS(lr.predict(pred), f_y_test))

            # print & save the model
            # print("stack loss:", loss_list[j + 1][-1])
            joblib.dump(filename="../temp_model/stack" + str(i), value=lr)
            i += 1

        print("=" * 50)
        for i in range(len(chose_model)):
            print(
                chose_model[i],
                (10 - len(chose_model[i])) * " " + ":",
                np.mean(loss_list[i]),
            )
        score.append(np.mean(loss_list[i + 1]))
        print("stack", (10 - len("stack")) * " " + ":", np.mean(loss_list[i + 1]))
        print("=" * 79)

        plt.subplots(figsize=(9, 9))
        sns.heatmap(
            pd.DataFrame(
                np.c_[pred, lr.predict(pred)], columns=([chose_model + ["stack"]])
            ).corr(),
            annot=True,
            vmax=1,
            square=True,
            cmap="Blues",
        )

        #####################################################################################################
        # Predict

        # stacking with stack model
        if scale:
            d_test = scaler.transform(d_test)
        prediction = []
        for i in range(splits):
            isFirst = True
            for model_name in chose_model:
                model = joblib.load(filename="../temp_model/{}{}".format(model_name, i))
                if isFirst:
                    pred = model.predict(d_test)
                    isFirst = False
                else:
                    pred = np.c_[pred, model.predict(d_test)]
            prediction.append(lr.predict(pred))

        res_pred = np.mean(np.array(prediction), axis=0)

        # models corr matrix
        # models
        plt.subplots(figsize=(9, 9))
        sns.heatmap(
            pd.DataFrame(
                np.c_[pred, lr.predict(pred)], columns=([chose_model + ["stack"]])
            ).corr(),
            annot=True,
            vmax=1,
            square=True,
            cmap="Blues",
        )

        # stacking model
        plt.subplots(figsize=(9, 9))
        sns.heatmap(
            pd.DataFrame(np.array(prediction).T).corr(),
            annot=True,
            vmax=1,
            square=True,
            cmap="Blues",
        )

        all_pred = pd.DataFrame([group, res_pred], index=["Group", catg]).T

        mean_pred = [
            np.mean(all_pred[all_pred["Group"] == i][catg]) for i in range(120)
        ]
        cov_pred = [np.cov(all_pred[all_pred["Group"] == i][catg]) for i in range(120)]

        #####################################################################################################
        # Save

        # clean
        if catg == target[0]:
            pred = pd.read_csv(path["submit"], encoding="gbk")
            for t in target:
                pred[t + " ratio"] = np.zeros(120)
            pred.to_csv(s_res, index=False)

        # save this part
        pred = pd.read_csv(s_res, encoding="gbk")
        pred[catg + " ratio"] = mean_pred
        pred.to_csv(s_res, index=False)

        # save the log
        if catg == target[3]:
            seed_log = pd.read_csv(path["res"] + "/seed_log.csv")
            seed_log = seed_log.append([{"seed": seed, "score": np.mean(score)}])
            seed_log.to_csv(path["res"] + "/seed_log.csv", index=False)

#####################################################################################################
# stack seed
seed_log = pd.read_csv(path["res"] + "/seed_log.csv")
pred = []
for s in seed_log["seed"]:
    pred.append(pd.read_csv(path["res"] + "/submit_{}.csv".format(s)))

count = len(pred)
mean_res = pd.DataFrame(index=[i for i in range(120)], columns=pred[0].columns)
for i, r in enumerate(pred):
    for t in target:
        if i == 0:
            mean_res[t] = r[t] / count
        else:
            mean_res[t] = mean_res[t] + r[t] / count
    print(i, count)
mean_res["Group"] = pred[0]["Group"]
mean_res.to_csv(path["res"] + "/submit_mean.csv", index=False)
