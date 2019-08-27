# Awsome-CCF2019-kernals
收集CCF2019`离散制造过程中典型工件的质量符合率预测`赛道上，我看见的一些public kernals

**当前kernal代码都来自可公开访问的网站，如果你并不想某个kernal出现在这里，请直接issue,我会立即移除**

|kernals|重构|源地址|
|---|---|---|
|[lgbm种子融合](./lgbm_seed_stack.py)||https://zhuanlan.zhihu.com/p/79687336|
|[lgbm单模型](./lgbm.py)||https://github.com/destiny19960207/CCF_BDCI2019_discrete-manufacturing/blob/master/baseline.py|

我对这些public kernals进行了:
 - 重构部分代码
 - 按我的习惯重命名变量
 - 添加部分注释

我的修改**有可能**会提升代码可读性或效率，但并不确定具有任何优化效果. 修改是我在读别人的kernal时顺手完成的，其根本目的是确保我正确理解了代码执行过程和作者思路．

我的习惯命名(前缀中缀或后缀):
 - ```d_``` 数据集
 - ```x_``` 输入
 - ```y_``` 输出(目标)
 - ```_train``` 训练
 - ```_test``` 测试
 - ```_valid``` 验证
 - ```_pred``` 预测
 - ```_index_``` 某组索引
 - 其他单字母作为第一前缀: 某个循环内部的临时变量，单字母表示当前循环变量
