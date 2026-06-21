
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from mlxtend.regressor import StackingCVRegressor
from sklearn.model_selection import train_test_split
import torch.optim as optim
from skopt import BayesSearchCV
from skorch import NeuralNetRegressor
from skorch.callbacks import Checkpoint
# from skorch.dataset import CVSplit
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import TimeSeriesSplit


# 创建一个数据集
data_origin = pd.read_csv(r"C:\Users\86150\Desktop\集成学习\集成学习代码\集成学习代码\第四轮单井筛选数据集.csv")
data_origin = data_origin.iloc[0:360, 1:]

data_origin = pd.DataFrame(data_origin)
# 分割特征和目标变量
X_select = data_origin.drop(["oilpro-lab"], axis=1)
print(X_select)
y = data_origin["oilpro-lab"]

print("X_select形状的大小：")
print(X_select.shape)
print("X_select的属性：" + str(type(X_select)))
# X_exam = X_select.iloc[:, [1, 2, 3, 10, 11, 12, 13, 14]]
X_exam = X_select.iloc[:, [1, 2 , 10,  12, 13, 14]]
y_exam = y

# 加载训练数据和标签
X_train, X_test, y_train, y_test = train_test_split(
    X_exam, y_exam, test_size=0.2, random_state=42)



# 定义 TCN 模型
class TCN(nn.Module):
    def __init__(self, input_size=X_train.shape[1], output_size=1, hidden_units=60, dropout=0.2):
        super(TCN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

tcn_model = NeuralNetRegressor(
    module = TCN,
    module_output_size = 1,
    module_dropout = 0.2,
    criterion=torch.nn.MSELoss,
     optimizer=optim.Adam,
    optimizer__lr = 0.01,
    max_epochs=180,
    verbose=False
)

# 定义随机森林、XGBoost和TCN模型
rf_model = RandomForestRegressor()


# 创建基学习器列表
base_models = [rf_model,  tcn_model]


# 创建元学习器模型
meta_model = LinearRegression()

# 使用 TimeSeriesSplit 创建交叉验证
tscv = TimeSeriesSplit(n_splits=5)
# 创建 stacking 集成学习模型
stacking_model = StackingCVRegressor(regressors=base_models, meta_regressor=meta_model, cv=KFold(n_splits=10))

# 参数列表
param_grid = {
    # TCN
    'neuralnetregressor__hidden_size':(12,200),
    'optimizer__lr': (0.001 , 0.8),
    # RF
    "randomforestregressor__n_estimators": (6, 100),
    "randomforestregressor__max_depth": (1, 16),
    "randomforestregressor__min_samples_split": (2, 16),
    "randomforestregressor__min_samples_leaf": (1, 16),

}

grid_search = BayesSearchCV(
    stacking_model, param_grid, scoring="r2", cv=5,
)  # cv=5表示交叉验证5次，scoring='r2'表示以R-squared作为模型评价准则

# 输出参数最优值
grid_search.fit(X_train, y_train)  # 传入数据
best_params = grid_search.best_params_  # 输出参数的最优值
best_score = grid_search.best_score_
print("Best Params:", best_params)
print("Best score:", best_score)

best_model = StackingCVRegressor(**best_params)
# 在测试集上进行预测
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# 输出随机森林回归的均方误差和决定系数
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
# 计算平均绝对百分比误差（MAPE）
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
# print(mape)

print("随机森林回归均方误差：", mse)
print("随机森林回归决定系数：", r2)
print("随机森林回归平均误差：", mae)
print("LGBM回归根方误差：", rmse)
# print("Extra-Trees回归平均绝对百分比误差：", mape)


'''
# 训练 stacking 模型
stacking_model.fit(X_train, y_train)

# 加载测试数据
X_test = X_test

# 使用 stacking 模型进行预测
predictions = stacking_model.predict(X_test)

# 打印预测结果
print(predictions)

# 计算指标
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("MSE:", mse)
print("R2:", r2)
'''
