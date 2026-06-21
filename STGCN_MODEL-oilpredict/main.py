# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from torchsummary import summary

from STGCNMODEL import GCN_model
from data_utils import generate_dataset1, generate_dataset2, load_metr_la_data, get_normalized_adj  # 产生数据，引入数据，标准化邻接矩阵。

# from pyswarm import pso
# from sklearn.metrics import make_scorer, fbeta_score, accuracy_score
# from sklearn.model_selection import GridSearchCV, KFold
# from sko.PSO import PSO

use_gpu = False
num_timesteps_input = 16  # 调整时间步数
num_timesteps_output = 1  # 调整产出时间步数
split = 0.8
kernel_size = 4
epochs = 40
batch_size = 1
time_sqeeze = 0  # 时滞：time_sqeeze + 1
learning_rate = 0.000001
dropout1 = 0
dropout2 = 0
dropout3 = 0
L2 = 0

parser = argparse.ArgumentParser(description='STGCN')  # 调用计算单元
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()
args.device = None
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


def train_epoch(input_inji, target_inji, batch_size, L2):  # 计算每一轮损失
    # def train_epoch(input_inji, target_inji, input_oil, batch_size):  # 计算每一轮损失
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(input_inji.shape[0])  # 括号内数字以下的所有排列组合，0 - n-1 个，整个数据集遍历
    epoch_training_losses = []
    optimizer = torch.optim.Adam(net.parameters(), weight_decay=L2, lr=learning_rate)  # weight_decay:L2正则化参数
    for i in range(0, input_inji.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()  # 初始化梯度为零

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = input_inji[indices], target_inji[indices]  # 特征标签的批次输入量
        # x_add = input_oil[indices]                                            # 视时滞情况选择性加入
        # print("X_batch的形状大小为："+str(X_batch.shape),"y_batch的形状大小为："+str(y_batch.shape),"x_add的形状大小为："+str(x_add.shape))
        X_batch = X_batch.to(device=args.device)  # 将变量输入设备
        y_batch = y_batch.to(device=args.device)  # 将特征输入设备
        # x_add = x_add.to(device=args.device)
        # out = net(A_wave, X_batch, x_add)                                     # todo:是否训练模型的意思  视情况选择性加入
        out = net(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)  # todo；
        # print(loss.shape)
        loss.backward()  # 梯度下降
        optimizer.step()  # 参数更新
        epoch_training_losses.append(loss.detach().cpu().numpy())  # 把每轮的标签损失取出来
        # print(np.array(epoch_training_losses).shape)
    return sum(epoch_training_losses) / len(epoch_training_losses)  # 返回每个batch_size的损失函数的值


if __name__ == '__main__':
    torch.manual_seed(42)

    A, X, means, stds = load_metr_la_data()  # 从数据模块载入数据 , X的数据结构为
    print("原始数据：" + str(X.shape))
    X_test = X.copy()

    X1 = X_test
    print(X1)
    print(X1.dtype)
    split_line1 = int(X1.shape[2] * split)  # X形状为：6,1,380，380*0.6
    split_line2 = int(X1.shape[2] * 1)  # 380*0.8

    train_oil_data = X1[:, :,
                     :split_line1]  # 此处 train_oil_data     代表对原始动态数据集划分，图方便           todo:将这一步嵌入时滞模块当中                                  # 注水验证集数据
    val_oil_data = X1[:, :, split_line1:split_line2]  # 注水验证集数据
    test_oil_data = X1[:, :, split_line2:]

    training_oil_input, training_oil_target = generate_dataset2(train_oil_data,
                                                                # 训练集，此时数据结构为：（8，时间步数，1，样本集个数）  样本集个数= oil_data - 15
                                                                num_timesteps_input=num_timesteps_input,
                                                                num_timesteps_output=num_timesteps_output)
    print("采油切片数据训练集输入的形状：" + str(
        training_oil_input.shape))  # 测试一下形状：[586, 8, 12, 1] B,H,W,C          样本集个数= oil_data - 15
    print("采油切片数据训练集第一维输入的形状：" + str(training_oil_input.shape[0]))  # 训练集第一维维度为586
    val_oil_input, val_oil_target = generate_dataset2(val_oil_data,
                                                      num_timesteps_input=num_timesteps_input,  # 验证集
                                                      num_timesteps_output=num_timesteps_output)
    print(
        "采油切片数据验证集输入的形状：" + str(val_oil_input.size()))  # 验证集的形状：torch.Size([185, 8, 12, 1])       样本集个数= oil_data - 15
    test_oil_input, test_oil_target = generate_dataset2(test_oil_data,  # 测试集
                                                        num_timesteps_input=num_timesteps_input,
                                                        num_timesteps_output=num_timesteps_output)
    print("采油切片数据测试集输入的形状：" + str(test_oil_input.size()))  # 测试集的形状：torch.Size([185, 8, 12, 1])

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)
    A_wave = A_wave.to(torch.float32)

    A_wave = A_wave.to(device=args.device)  # 将参数放置到设备当中
    net = GCN_model(A_wave.shape[0],  # 8 个点
                training_oil_input.shape[3],  # 对照无时滞模块
                hidden1_out=64,
                out_channels2=64,
                kernel_size=kernel_size,
                spatial_channels=16,  # todo:空间节点个数选择 12
                hidden3_out=64,
                hidden4_in=64,
                num_timesteps_input=num_timesteps_input,  # 输入的时间步
                num_timesteps_output=num_timesteps_output,
                dropout1=dropout1,
                dropout2=dropout2,
                dropout3=dropout3).to(device=args.device)  # 时间步输出

    epoch_training_losses = []
    optimizer = torch.optim.Adam(net.parameters(), weight_decay=L2, lr=learning_rate)  # weight_decay:L2正则化参数
    loss_criterion = nn.MSELoss()
    out = net(A_wave, training_oil_input)
    loss = loss_criterion(out, training_oil_target)  # todo；
    # print(loss.shape)
    loss.backward()  # 梯度下降
    optimizer.step()  # 参数更新
    epoch_training_losses.append(loss.detach().cpu().numpy())  # 把每轮的标签损失取出来

    train_losses = []
    val_losses = []
    train_predicts = []
    val_maes = []
    val_r2 = []
    val_rmses = []
    val_target_whole = []  # 所有的验证集原始值目标存储
    val_prediction_whole = []  # 所有的验证集预测值目标存储
    with torch.no_grad():  # 数据不需要梯度计算，即不会进行反相传播
        net.eval()  # 不加的话即使没有训练输入数据也会改变权值，因为这是禁止forward过程对参数造成的影响
        # val_input = val_input.to(device=args.device)
        # val_target = val_target.to(device=args.device)

        out = net(A_wave, val_oil_input)
        # print(out.shape)     # 形状为 整个测试集的个数
        val_loss = loss_criterion(out, val_oil_target).to(device="cpu")  # 验证集损失函数收集
        #
        val_losses.append(val_loss.detach().numpy().item())  # np.asscalar转换为np.item()
        out_unnormalized = out.detach().cpu().numpy() * stds[0] + means[0]  #
        target_unnormalized = val_oil_target.detach().cpu().numpy() * stds[0] + means[0]  #
        # print(out_unnormalized.shape)
        val_prediction_whole.append(out_unnormalized)
        # print(type(val_prediction_whole))                                                    # class list
        val_target_whole.append(target_unnormalized)
        # print(np.absolute(out_unnormalized - target_unnormalized))
        # print(np.absolute(out_unnormalized - target_unnormalized).shape)
        mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))  #
        # val_maes.append(mae)                                                                  #

        out = None

        val_oil_input = val_oil_input.to(device="cpu")  #
        val_oil_target = val_oil_target.to(device="cpu")  #

    # for epoch in range(epochs):
    #     loss = train_epoch(training_oil_input, training_oil_target,batch_size=batch_size,L2=L2)       # 返回损失训练模型
    #     train_losses.append(loss)
    #     # Run validation
    #     with torch.no_grad():                                                                    # 数据不需要梯度计算，即不会进行反相传播
    #         net.eval()                                                                           # 不加的话即使没有训练输入数据也会改变权值，因为这是禁止forward过程对参数造成的影响
    #         # val_input = val_input.to(device=args.device)
    #         # val_target = val_target.to(device=args.device)
    #
    #         out = net(A_wave,val_oil_input)
    #         # print(out.shape)     # 形状为 整个测试集的个数
    #         val_loss = loss_criterion(out, val_oil_target).to(device="cpu")                       # 验证集损失函数收集
    #                                                                                             #
    #         val_losses.append(val_loss.detach().numpy().item())                                   # np.asscalar转换为np.item()
    #         out_unnormalized = out.detach().cpu().numpy()*stds[0]+means[0]                        #
    #         target_unnormalized = val_oil_target.detach().cpu().numpy()*stds[0]+means[0]          #
    #         # print(out_unnormalized.shape)
    #         val_prediction_whole.append(out_unnormalized)
    #         # print(type(val_prediction_whole))                                                    # class list
    #         val_target_whole.append(target_unnormalized)
    #         # print(np.absolute(out_unnormalized - target_unnormalized))
    #         # print(np.absolute(out_unnormalized - target_unnormalized).shape)
    #         mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))                    #
    #         val_maes.append(mae)                                                                  #
    #
    #         out = None
    #
    #         val_oil_input = val_oil_input.to(device="cpu")  #
    #         val_oil_target = val_oil_target.to(device="cpu")                                      #

    # r2 = r2_score(np.array(validation_prediction)[:,0],np.array(validation_target)[:,0])
    # MSE = mean_squared_error(np.array(validation_prediction)[:,0],np.array(validation_target)[:,0])
    # rMSE = np.sqrt(MSE)
    # print('Validation r2:'+ str(r2))
    # print('Validation rMSE:'+str(rMSE))

    # 打印预测值
    # print(type(val_target_whole))
    # print(val_target_whole)
    # print(np.array(val_prediction_whole).shape)                   #  (epoch,test_len,label)
    val_prediction = np.array(val_prediction_whole)[-1, :, :].reshape(-1, 1)
    # print(val_prediction)
    print(val_prediction.shape)
    # val_prediction = np.array(val_prediction_whole)[-1,:,:].reshape(-1,1)
    # val_prediction_list = list(val_prediction)
    val_target = np.array(val_target_whole)[-1, :, :].reshape(-1, 1)
    # print(val_target.shape)
    # print(val_target)
    val_tar = pd.DataFrame(val_target)
    # val_tar.to_csv(r'C:\Users\anki\Desktop\val_tar.csv')
    # print(val_tar.shape)
    # print(val_tar)

    val_target_list = list(val_target)
    print(val_target_list)
    val_prediction_list = list(val_prediction)

    # print(val_target)
    # print(val_maes)
    # print(val_losses)
    mse = np.mean(val_losses)
    rMSE = np.sqrt(mse)
    # r2_score = r2(torch.from_numpy(val_target), torch.from_numpy(val_prediction))

    print('validation_prediction的形状为:' + str(np.array(val_prediction).shape))
    print('Training loss的形状:' + str(np.array(train_losses).shape))
    # print(np.array(val_target))
    print('validation_target的形状：' + str(np.array(val_target).shape))
    # print(val_prediction)
    # print(val_target)                  # 验证集目标值一致
    print('validation_target的形状：' + str(val_oil_target.shape))
    print("Training loss: {}".format(train_losses))
    print("Validation loss: {}".format(val_losses))
    print("Validation MAE: {}".format(val_maes))
    print("Validation rMSE: {}".format(rMSE))
    print('r2_score:{}'.format(r2_score))

    # plt.figure(dpi=300)
    plt.plot(train_losses, label="training loss")
    plt.plot(val_losses, label="validation loss")
    plt.legend()
    plt.show()

    plt.plot(val_prediction, label="validation_prediction")
    plt.plot(val_target, label="validation_target")
    plt.legend()
    plt.show()

    # plt.figuresize(figuresize=(10,5),dpi='600')

    checkpoint_path = "checkpoints/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    with open("checkpoints/losses.pk", "wb") as fd:
        pk.dump((train_losses, val_losses, val_maes), fd)


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))


'''
#   参数寻优
    # 定义损失函数
    loss_fn = nn.MSELoss()

    # 定义适应度函数
    def fitness_function(params, model, X, y):
        # 更新神经网络参数
        # for i, param in enumerate(model.parameters()):
        #     param.data.copy_(torch.tensor(params[i]))
    # 初始化参数
        net = STGCN(A_wave.shape[0],  # 8 个点
                    training_oil_input.shape[3],  # 对照无时滞模块
                    hidden1_out=12,
                    out_channels2=12,
                    kernel_size=kernel_size,
                    spatial_channels=8,  # todo:空间节点个数选择
                    hidden3_out=6,
                    hidden4_in=6,
                    num_timesteps_input=num_timesteps_input,  # 输入的时间步
                    num_timesteps_output=num_timesteps_output,
                    dropout1=dropout1,
                    dropout2=dropout2)
        # num_nodes = A_wave.shape[0]
        # num_features = training_oil_input.shape[3]

        # 训练神经网络
        train_epoch(net, X, y,L2, learning_rate)
        # 计算损失值（适应度）
        output = model(X)
        loss = loss_fn(output, y)
        return loss.item()
    # 定义模型
    # todo；此处省略实例化模型
    # 定义参数上下限
    lower_bound = [6, 6, 2,4,4,4,8,2,0.1,0.1,]    # 参数下限，有几个参数就有对应的参数界限值
    upper_bound = [15, 15, 5,10,8,8,15,4,1,1]

    # 使用PSO优化TCN模型参数
    best_params, best_loss = pso(fitness_function, lower_bound, upper_bound, args=(net, training_oil_input, training_oil_target))

    # 更新超参数为最优值
    learning_rate = best_params[0]
    kernel_size = int(best_params[1])
    dropout = best_params[2]
    net.conv1.kernel_size = (kernel_size,)
    net.conv2.kernel_size = (kernel_size,)
    net.dropout.p = dropout

    # 训练模型
    train_epoch(net, training_oil_input, training_oil_target, learning_rate)

    # 输出最优结果
    output = net(training_oil_input)
    print("最优结果：", output)

'''

'''
# 粒子群搜索算法

    # 定义适应度函数（即损失函数）
    def fitness_function(params, model, X, y):
        # 更新神经网络参数
        for i, param in enumerate(model.parameters()):
            param.data.copy_(torch.tensor(params[i]))

        # 训练神经网络
        train(model, X, y)

        # 计算损失值（适应度）
        output = model(X)
        loss = loss_fn(output, y)
        return loss.item()


    # 定义输入和输出数据
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    # 定义神经网络模型
    input_size = X.shape[1]
    hidden_size = 5
    output_size = y.shape[1]
    model = Net(input_size, hidden_size, output_size)

    # 定义参数上下限
    lower_bound = [-5.0] * sum(p.numel() for p in model.parameters())
    upper_bound = [5.0] * sum(p.numel() for p in model.parameters())

    # 使用PSO优化BP神经网络参数
    best_params, best_loss = pso(fitness_function, lower_bound, upper_bound, args=(model, X, y))

    # 更新神经网络参数为最优值
    for i, param in enumerate(model.parameters()):
        param.data.copy_(torch.tensor(best_params[i]))

    # 输出最优结果
    output = model(X)
    print("最优结果：", output)
'''

# 网格搜索算法
# param_grid = {'kernel_size':[2,3,4,5,6],
#               'batch_size':[10,15,20],
#               'epochs':[30,20,15],
#               'lr':np.arange(0,1),
#               'dropout1':np.arange(0,1,0.1),
#               'dropout2':np.arange(0,1,0.1),
#               'weight_decay':np.arange(0,1,0.1)}
#
# grid_search = GridSearchCV(estimator = net(A_wave,training_oil_input),
#                                      param_grid = param_grid,
#                                      scoring='r2',
#                                      n_jobs=4,
#                                      cv=None)
# grid_result = grid_search.fit(training_oil_input, training_oil_target)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


'''
import numpy as np

# 定义扩展卡尔曼滤波器类
class ExtendedKalmanFilter:
    def __init__(self, f, h, jF, jH, Q, R, x0, P0):
        self.f = f  # 状态转移函数
        self.h = h  # 观测函数
        self.jF = jF  # 状态转移函数的雅克比矩阵
        self.jH = jH  # 观测函数的雅克比矩阵
        self.Q = Q  # 状态转移噪声协方差矩阵
        self.R = R  # 观测噪声协方差矩阵
        self.x = x0  # 状态向量
        self.P = P0  # 状态协方差矩阵

    def predict(self, u):
        # 预测状态向量
        self.x = self.f(self.x, u)

        # 预测状态协方差矩阵
        self.P = np.dot(np.dot(self.jF(self.x), self.P), self.jF(self.x).T) + self.Q

        return self.x, self.P

    def update(self, z):
        # 计算卡尔曼增益矩阵
        K = np.dot(np.dot(self.P, self.jH(self.x).T),
                   np.linalg.inv(np.dot(np.dot(self.jH(self.x), self.P), self.jH(self.x).T) + self.R))

        # 更新状态向量和状态协方差矩阵
        self.x = self.x + np.dot(K, (z - self.h(self.x)))
        self.P = np.dot((np.identity(self.P.shape[0]) - np.dot(K, self.jH(self.x))), self.P)

        return self.x, self.P


其中，`f`和`h`分别表示状态转移函数和观测函数，`jF`和`jH`分别表示状态转移函数和观测函数的雅克比矩阵，`Q`和`R`分别表示状态转移噪声协方差矩阵和观测噪声协方差矩阵，`x0`和
`P0`分别表示初始状态向量和初始状态协方差矩阵。`predict`方法用于预测状态向量和状态协方差矩阵，`update`方法用于根据观测值更新状态向量和状态协方差矩阵。

需要注意的是，在使用扩展卡尔曼滤波算法时，需要根据具体的问题进行状态转移函数和观测函数的定义，以及雅克比矩阵的计算，以确保算法的正确性。


import torch
# 定义一维卷积网络
net = torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
# 输出网络参数的权重
print('weights: ', net.weight.data)
# 输出网络参数的偏置系数
print('biases: ', net.bias.data)
'''
