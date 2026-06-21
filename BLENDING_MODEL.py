from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor , IsolationForest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error , mean_absolute_percentage_error
from skorch import NeuralNetRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression,Lasso
import torch.optim as optim
import matplotlib.pyplot as plt
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import TimeSeriesSplit
import random
import torch
import torch.nn as nn
from sklearn.svm import SVR
# from sklearn.datasets.samples_generator import make_blobs


'''创建训练的数据集'''
# data, target = make_blobs(n_samples=50000, centers=2, random_state=0, cluster_std=0.60)    # 生成测试数据集

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # 读取无处理数据
# data_origin = pd.read_csv(r"C:\Users\anki\Desktop\集成学习8.9\集成学习\集成学习代码\集成学习代码\第四轮单井筛选数据集.csv")
#
# data_origin = data_origin.iloc[0:360, 1:]
# data_origin = pd.DataFrame(data_origin)
# X_select = data_origin.drop(["oilpro-lab"], axis=1)
# y = data_origin["oilpro-lab"]
# # X_exam = X_select.iloc[:, [1,4, 10, 12, 13, 14,15,20]]
# #  油井自身历史序列数据
# X_exam = X_select.iloc[:, 6:16]
# y_exam = y


# 2006.01 - 2021.01数据 # todo: 抽油井自身数据实验
# data_origin = pd.read_csv(r"C:\Users\anki\Desktop\2006.01-2021.01KNN填补水井与油井后数据.csv")
data_origin = pd.read_csv(r"C:\Users\anki\Desktop\2006.01-2021.01 大论文最终章记录数据\blending小论文数据集.csv")
data_origin = pd.DataFrame(data_origin)
columns_select = ['BengJing' , 'ChongCi' ,'Dongyemian' , "oilpro-0" , "oilpro-lab"]
# 'kalman-oil-0',	'kalman-oil-lab' ,	'kalman-water-0' ,'kalman-water-lab',"oilpro-0","oilpro-lab","waterpro-0",'waterpro-lab',
X_select = data_origin[columns_select]
X_select = X_select.dropna()
print(X_select)

X_select = data_origin[columns_select]
X_select = X_select.dropna()
print(X_select)


# 定义数据集
def create_sequences(data, seq_length):
    xs = []
    # ys = []
    for i in range(len(data) - seq_length ):
        x = data[i:(i + seq_length)]
        # y = data[(i + seq_length)]
        xs.append(x)
        # ys.append(y)
    return np.array(xs)


seq = 1
y = X_select["oilpro-lab"]
X_exam = X_select.drop(["oilpro-lab"], axis=1)
X_SEQ = create_sequences(X_exam , seq)
X_exam = X_SEQ.reshape(X_SEQ.shape[0] , -1)
print(X_exam.shape)
y = y.iloc[seq:]
print(y)
print("y形状")
print(y.shape)

#  油井自身历史序列数据
lag = 0
X_exam = X_exam[:(len(X_exam)-lag),:]
y = y.iloc[lag:]


# 归一化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(np.array(X_exam))
y_scaled = scaler.fit_transform(np.array(y).reshape(-1, 1))
print(X_scaled.shape)
print(y_scaled.shape)

# todo:划分数据集，满足测试及预测要求
split_index = len(X_exam) - 12
print(len(X_exam),len(y))
X_tra , X_test = X_scaled[:split_index], X_scaled[split_index:]
y_tra , y_test = y_scaled[:split_index], y_scaled[split_index:]
X_train, X_val, y_train , y_val = train_test_split(X_tra ,  y_tra , test_size = 0.1 , shuffle = True , random_state = 42 )
# shuffle = False , random_state = None , shuffle = True , random_state = 42
# # 设置随机数种子 todo:混乱训练集
# random.seed(42)
# # 使用 np.random.shuffle() 打乱数组
# np.random.shuffle(X_train)
# np.random.shuffle(y_train)

print(X_tra.shape,X_test.shape,
      y_tra.shape,y_test.shape)
print(X_train.shape,X_test.shape,
      y_train.shape,y_test.shape)


# 将数据转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val , dtype=torch.float32).to(device)
X_test = torch.tensor( X_test,dtype=torch.float32).to(device)
y_train = torch.tensor(y_train,dtype=torch.float32).to(device)
y_val = torch.tensor(y_val , dtype=torch.float32).to(device)
y_test = torch.tensor(y_test,dtype=torch.float32).to(device)

# 定义TCN模型
class TCN(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_units= 32 , dropout=0.2):
        super(TCN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        x = x.clone().detach().requires_grad_(True)
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze()

tcn_model = NeuralNetRegressor(
    module=TCN,
    module__input_size = X_train.shape[1] ,
    module__output_size = 1,
    module__hidden_units = 16 ,     #  29
    module__dropout=0.2 ,
    criterion=torch.nn.MSELoss ,
    optimizer=optim.Adam ,
    optimizer__lr = 0.000001,
    max_epochs = 1600 ,
    verbose=False
)

grdb_model = GradientBoostingRegressor(

)

# '''模型融合中使用到的各个单模型'''
clfs = [RandomForestRegressor(
                              # n_estimators=180 , max_depth= 3 , min_samples_split=3 ,
                              # min_samples_leaf = 3 ,
                              ),
        grdb_model ,
        tcn_model
        # RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
        # ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        # GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=5)
        ]

# 切分一部分数据作为测试集
# X, X_predict, y, y_predict = train_test_split(data, target, test_size=0.33 , random_state = 42 )

#  todo:stacking运用下的划分数据集方法，有新版本可以用
# n_folds = 5
# skf = list(StratifiedKFold(y, n_folds))
# skf = list(StratifiedKFold(n_splits=n_folds).split(X_train, y_test))

# '''切分训练数据集为d1,d2两部分'''
# X_d1, X_d2, y_d1, y_d2 = train_test_split(X_train, y_train, test_size=0.3 , random_state=2017)
X_d1, X_d2, y_d1, y_d2 = X_train, X_val, y_train , y_val
dataset_d2 = np.zeros((X_d2.shape[0], len(clfs)))     #  元学习器的空训练集，即中间验证集的空集
dataset_d3 = np.zeros((X_test.shape[0], len(clfs)))   #  元学习器的空测试集，即最终测试集的空集
dataset_d1 = np.zeros((X_d1.shape[0], len(clfs)))
# dataset_d2 = np.zeros((X_predict.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    '''依次训练各个单模型'''
    print(j, clf)
    '''使用第1个部分作为预测，第2部分来训练模型，获得其预测的输出作为第2部分的新特征。'''
    # X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
    y_d1 = y_d1.ravel()
    clf.fit(X_d1, y_d1)        #  基学习器训练
    y_submission = clf.predict(X_d2)  #  中间过度的元学习器的训练集预测，即整体数据的中间验证集预测
    # y_submission = scaler.inverse_transform(y_submission.reshape(-1, 1))

    y_submission = y_submission.ravel()
    dataset_d2[:, j] = y_submission
    '''对于测试集，直接用这k个模型的预测值作为新的特征。'''
    dataset_d3[:, j] = clf.predict(X_test)  #  元学习器的测试集预测，即中间验证集
    dataset_d1[:,j] = clf.predict(X_d1)
    # print("val auc Score: %f" % roc_auc_score(X_test, dataset_d2[:, j]))


'''融合使用的模型'''  #  元学习器的训练
# clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
clf = LinearRegression()
# clf = SVR()
# clf = BayesianRidge()
# clf = Lasso()

# 使用元模型训练中间过渡数据
clf.fit(dataset_d2, y_d2)


# print("Linear stretch of predictions to [0,1]")
# y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())     # 原模型的归一化
print("blend result")

y_test_pred = clf.predict(dataset_d3)   # 最终测试集预测效果
y_val_pred = clf.predict(dataset_d2)  # 验证集预测效果
y_train_pred = clf.predict(dataset_d1)  # 最初训练集预测效果

# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred)
y_train = scaler.inverse_transform(y_train)
y_val_pred = scaler.inverse_transform(y_val_pred)
y_val = scaler.inverse_transform(y_val)
y_test_pred = scaler.inverse_transform(y_test_pred)
y_test = scaler.inverse_transform(y_test)

print('y_test_pred')
print(y_test_pred.reshape(-1,1))

print('y_test')
print(np.array(y_test).reshape(-1,1))

print('y_train_pred')
print(np.array(y_train_pred).reshape(-1,1))

print('y_train')
print(np.array(y_train).reshape(-1,1))

print('y_val_pred')
print(np.array(y_train_pred).reshape(-1,1))

print('y_val')
print(np.array(y_train).reshape(-1,1))

# 计算平均绝对百分比误差（MAPE）
train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
# print(mape)
def mre(y_true, y_pred):
    """
    计算MRE指标
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: MRE值
    """
    mae = mean_absolute_error(y_true, y_pred)
    mre = mae / np.mean(y_true)
    return mre

train_mre = mre(y_train, y_train_pred)

# # 输出随机森林回归的均方误差和决定系数
# train_mse = mean_squared_error(y_train, y_train_pred)
# train_r2 = r2_score(y_train, y_train_pred)
# train_mae = mean_absolute_error(y_train, y_train_pred)
# train_rmse = np.sqrt(np.float_(train_mse))
# # 计算平均绝对百分比误差（MAPE）
# train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
# # print(mape)
# print("blending回归均方误差train_mse：",train_mse)
# print("blending回归决定系数train_r2：", train_r2)
# print("blending回归平均误差train_mae：", train_mae)
# print("blending回归平均绝对百分比误差 train_mape：", train_mape)


# 输出随机森林回归的均方误差和决定系数
val_mse = mean_squared_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = np.sqrt(np.float_(val_mse))
val_mre = mre(y_test, y_test_pred)
# 计算平均绝对百分比误差（MAPE）
val_mape = mean_absolute_percentage_error(y_val, y_val_pred)
print("blending回归均方误差val_mse：", val_mse)
print("blending回归决定系数val_r2：", val_r2)
print("blending回归平均误差val_mae：", val_mae)
print("blending回归平均绝对百分比误差 val_mape：", val_mape)
print("随机森林回归平均误差test_mre：", val_mre)


# 输出随机森林回归的均方误差和决定系数
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(np.float_(test_mse))
# 计算平均绝对百分比误差（MAPE）
test_mre = mre(y_test, y_test_pred)
# 计算平均绝对百分比误差（MAPE）
test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
print("blending回归均方误差test_mse：", test_mse)
print("blending回归决定系数test_r2：", test_r2)
print("blending回归平均误差test_mae：", test_mae)
print("blending回归平均绝对百分比误差 test_mape：", test_mape)
print("随机森林回归平均误差test_mre：", test_mre)

# 绘制原始数据和卡尔曼滤波后的数据
plt.figure(figsize=(8, 6) , dpi = 120)
# 设置字体
plt.rcParams['font.family'] = ['SimSun', 'Times New Roman'] # 设置字体族，中文为SimSun，英文为Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix' # 设置数学公式字体为stix
plt.plot(np.arange(len(y_test)) , y_test , label="real", color='tab:blue', alpha=0.6)
plt.plot(np.arange(len(y_test_pred)) , y_test_pred, label="predict", color='tab:orange')

# 获取当前的轴对象
ax = plt.gca()
# 选择刻度的间隔，非时间刻度
interval = 10
# 设置横坐标刻度的标签
ax.set_xticks(np.arange(len(y_test)))
# 设置横坐标刻度的标签文本
ax.set_xticklabels(np.arange(len(y_test)))

plt.legend()
plt.title("Singal well Water-productinon Predicting ", weight='black' , size=15)
plt.xlabel("Time (Day)", weight='black', labelpad= 5.0 , size=15)     # labelpad: 标签距离轴高度，默认 4.0
plt.ylabel("Water-production (T/Day)", labelpad= 5.0 , weight='black', size=15)   #
# plt.grid(True)
plt.show()



print('RF_model')
params = RandomForestRegressor().get_params()
for param, value in params.items():
    print(f"{param}: {value}")

print(' ')
print('grdb_model')
params = grdb_model.get_params()
for param, value in params.items():
    print(f"{param}: {value}")

print(" ")
print(f"{clf}")
print('stacking_Matal_SVR_parameters')
params= clf.get_params()
for param, value in params.items():
    print(f"{param}: {value}")

# params = clf.get_params()
# for param, value in params.items():
#     print(f"{param}: {value}")
