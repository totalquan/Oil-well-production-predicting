import numpy as np
import pandas as pd
import torch
# from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splrep, splev

#更改文件，邻接矩阵与数据集
def load_metr_la_data():          #   X 标准化完成

    # A = np.load("A.py.npy")                                                                        # A的原始8口井数据（模拟数据）
    A1 = [[1,0,0,1,1,1],[0,1,0,1,1,1],[0,0,1,1,1,1],[1,1,1,1,0,0],[1,1,1,0,1,0],[1,1,1,0,0,1]]       # A的6口井数据
    # A = [[1,0,0,0,1,1,1],[0,1,0,0,1,1,1],[0,0,1,0,1,1,1],[0,0,0,1,1,1,1],[1,1,1,1,1,0,0],[1,1,1,1,0,1,0],[1,1,1,1,0,0,1]]    # A的7口井数据

    # 7口井的数据量
    # A = [[1, 0, 0, 0, 1, 1, 1], [0, 1, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 1, 0], [1, 1, 1, 1, 0, 0, 1]]                      # A的7口井数据
    # #  采油井：1：X5-19 2：X5-181 3：X4N19 4：X6-161   注水井：5：X6-171 6：X5XN171 7：X5-C16
    # print(A.shape)
    # print(type(A))
    # X = pd.read_csv(r"C:\Users\anki\Desktop\油田日产量数据.csv")                                     # 转置后为（8，1，1000）
    # X = X.iloc[:, [4, 3, 2, 5, 10, 9, 8]].values                                                  # 注水量与产液量数据
    # #  采油井：1：X5-19 2：X5-181 3：X4N19 4：X6-161   注水井：5：X6-171 6：X5XN171 7：X5-C16
    # X = X.iloc[:, [14, 13, 12, 15, 10, 9, 8]].values                                              # 注水量与产油量数据
    # #  采油井：1：X5-19 2：X5-181 3：X4N19 4：X6-161   注水井：5：X6-171 6：X5XN171 7：X5-C16

    A = np.array(A1)
    X = pd.read_csv(r"C:\Users\anki\Desktop\cleaned_df.csv")                                      # 转置后为（6，1，380）
    X_test = X.iloc[:, [4, 2, 5, 10, 9, 8]].values                                                # 注水量与产液量数据
    X_test = X_test[:-21,:]
    # print(X_test)
    # X = X.iloc[:, [14, 12, 15, 10, 9, 8]].values                                                # 注水量与产油量数据
    # print(X.shape)
    # print(X[:,0])
    # X = X_test.transpose(1,0)
    # 定义箱线图检测异常值并进行超线性插值替换的函数
    # def replace_outliers(data, threshold=1):
    #
    #     for i in range(data.shape[0]):
    #         y = data[i, :]
    #
    #         # 计算第一维数据的中位数、第一四分位数和第三四分位数
    #         median = np.median(y)
    #         q1 = np.percentile(y, 25)
    #         q3 = np.percentile(y, 75)
    #
    #         # 计算离群值的阈值
    #         iqr = q3 - q1
    #         lower_threshold = q1 - threshold * iqr
    #         upper_threshold = q3 + threshold * iqr
    #
    #         # 检测异常值并进行超线性插值替换
    #         outliers = np.logical_or(y < lower_threshold, y > upper_threshold)
    #         if np.any(outliers):
    #             x_outliers = np.arange(y.shape[0])[outliers]
    #             y_outliers = y[outliers]
    #             spline = UnivariateSpline(x_outliers, y_outliers, k=3)
    #             y[outliers] = spline(x_outliers)
    #             # tck = splrep(x_outliers, y_outliers, k=1, s=0)
    #             # y[outliers] = splev(x_outliers, tck)
    #
    #     return data
    # X_test = replace_outliers(X)
    # X_clean = pd.DataFrame(X_test.transpose(1,0),columns=[4, 2, 5, 10, 9, 8])
    # X_clean.to_csv(r"C:\Users\anki\Desktop\X_clean.csv")
    #调用函数进行异常值检测和替换

    X_test = X_test[np.newaxis, :, :].transpose(2, 0, 1)                                                      # 添加第一维形状 ； 数组形状（1，380，6）
    # print(X_test)
    print('X_test的形状大小：'+str(X_test.shape))

    X1 = X_test.astype(np.float32)                                                                        # 将X转化为浮点数据
    # Normalization using Z-score method
    means = np.mean(X1, axis=(0, 2))                                            # 将 6 ， 380 的轴平均化
    X1 = X1 - means.reshape(1, -1, 1)                                            # todo: check
    stds = np.std(X1, axis=(0, 2))
    X = X1 / stds.reshape(1, -1, 1)                                             # 将 1，3 维度标准化

    return A, X, means, stds                                                   #  加载数据后的结构是 (6,1,380)

A, X, means, stds = load_metr_la_data()
print (X.shape)

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5                                                     # Prevent infs
    diag = np.reciprocal(np.sqrt(D))                                          # 将列表中的数据换为倒数
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset1(X, num_timesteps_input, num_timesteps_output,time_sqeeze):          # 建立数据集 X ，针对注水数据
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
# 添加展示机制
# X = np.load("spe10 case.npy").transpose((1, 2, 0))
# num_timesteps_input = 12, num_timesteps_output = 3
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
                                                                                # X形状为：6，1，380
    indices = [(i, i + (num_timesteps_input + num_timesteps_output+time_sqeeze)) for i    # 时间切片
               in range(X.shape[2] - (                                          # 遍历X的1000个数据轴，建立时间切片特征数据集，
                num_timesteps_input + num_timesteps_output)-time_sqeeze)]               # 遍历整个数据集产生，去掉末尾的多余一项
    # print(indices.shape())
    # Save samples
    features, target = [], []              # 设立特征与标签

# 原版代码
    # for i, j in indices:
    #     features.append(
    #         X[:, :, i: i + num_timesteps_input].transpose(
    #             (0, 2, 1)))
    #     target.append(X[:, 0, i + num_timesteps_input: j])

    for i,j in indices:                                                          #  遍历切片，产生特征与标签
        features.append(
            X[:, :, i: i + num_timesteps_input+time_sqeeze].transpose(                     #  转置前为6，1，21，转置过后为6，21，1；时间步数，1
                (0, 2, 1)))
        target.append(X[0, 0, i + num_timesteps_input+time_sqeeze: j])                     #  标签设置为：1，1，预测时间步长，

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))                                    #  时间步数切片

def generate_dataset2(X, num_timesteps_input, num_timesteps_output):             # 建立数据集 X，针对采油数据
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """

# 添加展示机制
# X = np.load("spe10 case.npy").transpose((1, 2, 0))
# num_timesteps_input = 12, num_timesteps_output = 3
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points

    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i        # 时间切片
               in range(X.shape[2] - (                                            # 遍历X的380个数据轴，建立时间切片特征数据集，
                num_timesteps_input + num_timesteps_output) )]                    # 遍历整个数据集产生
    # print(indices.shape())
    # Save samples
    features, target = [], []                                                     # 设立特征与标签

# 原版代码
    # for i, j in indices:
    #     features.append(
    #         X[:, :, i: i + num_timesteps_input].transpose(
    #             (0, 2, 1)))
    #     target.append(X[:, 0, i + num_timesteps_input: j])

    for i,j in indices:                                                          #  遍历切片，产生特征与标签
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(                       #  转置过后为：（6，1，12），时间步数，12
                (0, 2, 1)))
        target.append(X[0, 0, i + num_timesteps_input: j])                     #  标签设置为：1，1，预测时间步长

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))                                    #  时间步数切片

