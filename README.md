> **当前kernal代码都来自可公开访问的网站，如果你并不想某个kernal出现在这里，请直接issue,我会立即移除**
# Awesome-CCF2019-kernals
收集CCF2019`离散制造过程中典型工件的质量符合率预测`赛道上，我看见的一些public kernals

|kernals|重构|源地址|
|---|---|---|
|[lgbm种子融合](./lgbm_seed_stack.py)|[commit 3371a1259e2eb5383991ce7101da854ed4edccd8](https://github.com/loopyme/Awesome-CCF2019-kernals/commit/3371a1259e2eb5383991ce7101da854ed4edccd8)|https://zhuanlan.zhihu.com/p/79687336|
|[lgbm单模型](./lgbm.py)|[commit 5dd48af379adf3af2da6d01b66bfd1d24963f8b5](https://github.com/loopyme/Awesome-CCF2019-kernals/commit/5dd48af379adf3af2da6d01b66bfd1d24963f8b5)|https://github.com/destiny19960207/CCF_BDCI2019_discrete-manufacturing/blob/master/baseline.py|
|[万能一把梭](./model_smelter.py)|||
|[EDA工具](./Basic_EDA)|||

### 重构？
我对这些public kernals进行了:
 - 重构部分代码
 - 按我的习惯重命名变量
 - 添加部分注释

我的修改**有可能**会提升代码可读性或效率，但并**不确定具有任何优化效果**. 修改是我在读别人的kernal时顺手完成的，其根本目的是确保我正确理解了代码执行过程和作者思路．**能确定的是我重构后的代码会和原代码输出相同的结果**．

## 我的习惯命名
 - ```d_``` 数据集
 - ```x_``` 输入
 - ```y_``` 输出(目标)
 - ```_train``` 训练
 - ```_test``` 测试
 - ```_valid``` 验证
 - ```_pred``` 预测
 - ```_index_``` 某组索引
 - 其他单字母作为第一前缀: 某个循环内部的临时变量，单字母表示当前循环变量
