import torch
import torch.nn as nn
import numpy as np

"""
手动实现交叉熵

常用于分类任务
分类任务中，网络输出经常是所有类别上的概率分布

假设一个三分类任务，某样本的正确标签是第一类，则 p = [1, 0, 0],
模型预测值假设为[0.5, 0.4, 0.1]，则交叉熵计算如下：
H(p=[1,0,0], q=[0.5,0.4,0.1]) = -(1 * log0.5 + 0 * log0.4 + 0 * log0.1) ~= 0.3
p是真实值，q是预测值
q需要做归一化，否则无法使用交叉熵进行计算。需要保证交叉熵的值恒大于0，且当真实值和预测值相近的时候趋近于0（log的指数部分不能为0）
"""

#使用torch计算交叉熵（内部实现为：先对第一个参数调用softmax做归一化，再使用得到的结果与第二个参数做交叉熵的计算）
ce_loss = nn.CrossEntropyLoss()
#假设有3个样本，每个都在做3分类
pred = torch.FloatTensor([[0.3, 0.1, 0.3],
                          [0.9, 0.9, 0.9],
                          [0.5, 0.4, 0.2]])

#正确的类别分别为1, 2, 0
target = torch.LongTensor([1, 2, 0])
"""
[1, 2, 0]相当于如下
1 : [0, 1, 0]
2 : [0, 0, 1]
0 : [1, 0, 0]
"""

loss = ce_loss(pred, target)
print(loss, "torch输出交叉熵")

#实现softmax函数
def softmax(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix), axis=1, keepdims=True)

#验证softmax函数
print(torch.softmax(pred, dim=1))
print(softmax(pred.numpy()))

#将输入转化为onehot矩阵
def to_one_hot(target, shape):
    """
    此函数相当于做如下变换
    [1, 2, 0]相当于如下
    1 : [0, 1, 0]
    2 : [0, 0, 1]
    0 : [1, 0, 0]
    返回值为
    [[0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]]
    """
    one_hot_target = np.zeros(shape)
    for i, t in enumerate(target):            #[1, 2, 0] i为列表的下标， t为列表下标对应位置的值
        one_hot_target[i][t] = 1

    return one_hot_target

#手动实现交叉熵
def cross_entropy(pred, target):
    batch_size, class_num = pred.shape      #元组拆包
    pred = softmax(pred)                    #归一化
    target = to_one_hot(target, pred.shape)
    entropy = -np.sum(target * np.log(pred), axis=1)   #每一个样本都分别进行如下操作：向量的对位相乘再相加；最后 entropy = [[a], [b], [c]]
    return sum(entropy) / batch_size      #loss值累加再求平均

print(cross_entropy(pred.numpy(), target.numpy()), "手动实现交叉熵")
print(np.log(2.7))



















