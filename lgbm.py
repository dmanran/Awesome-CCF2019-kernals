import lightgbm as lgb
import pandas as pd

d_train = pd.read_csv("../data/first_round_training_data.csv")
d_test = pd.read_csv("../data/first_round_testing_data.csv")

########################################################################################################################
# ORI: 处理类别标签并分割x,y
#
# for i in range(1, 11):
#     par = "Parameter" + str(i)
#     tmp = d_train[par]
#     for j in range(6000):
#         x_train[j, i - 1] = tmp[j]
#         cls = cls2int[d_train["Quality_label"][j]]
#         y_train[j] = cls
# lgb_train = lgb.Dataset(x_train, y_train)

lgb_train = lgb.Dataset(
    d_train[["Parameter" + str(i) for i in range(1, 11)]],
    d_train["Quality_label"].map({"Excellent": 0, "Good": 1, "Pass": 2, "Fail": 3}),
)
########################################################################################################################

########################################################################################################################
# ORI: 丢弃Group信息
#
# for i in range(1, 11):
#     par = "Parameter" + str(i)
#     tmp = d_test[par]
#     for j in range(6000):
#         x_test[j, i - 1] = tmp[j]
#         ID = int(d_test["Group"][j])
#         test_id[j] = ID
x_test = d_test.drop(["Group"], axis=1)
########################################################################################################################

# lgbm params
params = {
    "boosting_type": "gbdt",
    "objective": "multiclassova",
    "num_class": 4,
    "metric": "multi_error",
    "num_leaves": 63,
    "learning_rate": 0.01,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_seed": 0,
    "bagging_freq": 1,
    "verbose": -1,
    "reg_alpha": 1,
    "reg_lambda": 2,
    "lambda_l1": 0,
    "lambda_l2": 1,
    "num_threads": 8,
}

# train
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=1300,
    valid_sets=[lgb_train],
    valid_names=["train"],
    verbose_eval=100,
)

# predict
y_pred = gbm.predict(x_test, num_iteration=1300)

########################################################################################################################
# ORI: 将结果丢进分组里,并整体缩放,保证概率和为1
#
# f = open("./lgb1300round.csv", "w")
# tmp = np.zeros([120, 4])
# cnt = np.zeros([120])
# for i in range(6000):
#     ID = test_id[i]
#     tmp[ID, :] += ans[i, :]
#     cnt[ID] += 1
# for i in range(120):
#     SUM = np.sum(tmp[i, :])
#     tmp[i, :] /= SUM
#
# f.write("Group,Excellent ratio,Good ratio,Pass ratio,Fail ratio\n")
# for i in range(120):
#     f.write(str(i))
#     for j in range(4):
#         f.write("," + str(tmp[i, j]))
#     f.write("\n")
# f.close()
y_pred = pd.DataFrame(
    y_pred, columns=["Excellent ratio", "Good ratio", "Pass ratio", "Fail ratio"]
)

y_pred["Group"] = d_test["Group"]
group_pred = y_pred.groupby("Group").mean()
for c in ["Excellent ratio", "Good ratio", "Pass ratio", "Fail ratio"]:
    group_pred[c] /= group_pred.sum(axis=1)
########################################################################################################################
group_pred.to_csv("../submit/lgb_submit.csv")
