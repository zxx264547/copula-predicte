from math import sqrt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy.io

# 读取数据
data = pd.read_excel('data_original.xlsx', header=None)
# 提取温度、辐照度和电流列的数据
temperature = data.iloc[:, 2]
radiation = data.iloc[:, 3]
current = data.iloc[:, 1]
# 创建MinMaxScaler的实例，用于归一化温度和辐照度
y_scaler = MinMaxScaler()
temperature_scaled = y_scaler.fit_transform(temperature.to_numpy().reshape(-1, 1))
radiation_scaled = y_scaler.fit_transform(radiation.to_numpy().reshape(-1, 1))
x_scaler = MinMaxScaler()
current_scaled = x_scaler.fit_transform(current.to_numpy().reshape(-1, 1))
# 将归一化后的温度和辐照度合并成输入特征
input_features = torch.tensor(list(zip(temperature_scaled, radiation_scaled)), dtype=torch.float32)


# 创建Bi-LSTM模型
class BiLSTMNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=50,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.out = nn.Sequential(
            nn.Linear(100, 1)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out


# 创建Bi-LSTM模型实例
input_size = 2  # 输入特征的数量，温度和辐照度
model_vine_copula = BiLSTMNet(input_size)
model_copula = BiLSTMNet(input_size)
model_one_step = BiLSTMNet(input_size)
model_quantile = BiLSTMNet(input_size)
model_original = BiLSTMNet(input_size)

# 加载保存的参数
model_vine_copula.load_state_dict(torch.load('module/model_parameters_vine_copula.pth'))
model_vine_copula.eval()  # 设置模型为评估模式
model_copula.load_state_dict(torch.load('module/model_parameters_copula.pth'))
model_copula.eval()  # 设置模型为评估模式
model_one_step.load_state_dict(torch.load('module/model_parameters_one_step.pth'))
model_one_step.eval()  # 设置模型为评估模式
model_quantile.load_state_dict(torch.load('module/model_parameters_quantile.pth'))
model_quantile.eval()  # 设置模型为评估模式
model_original.load_state_dict(torch.load('module/data_error.pth'))
model_original.eval()  # 设置模型为评估模式

# vine_copula进行预测
with torch.no_grad():
    predicted_current_vine_copula = model_vine_copula(input_features)
predicted_current_vine_copula[predicted_current_vine_copula < 0] = 0
length = len(input_features)
for i in range(length):
    if input_features[i, 1] == 0:
        predicted_current_vine_copula[i] = 0

# 反归一化预测结果
predicted_current_vine_copula = x_scaler.inverse_transform(predicted_current_vine_copula)
current = x_scaler.inverse_transform(current_scaled)

# 计算 RMSE
# current_scaled = np.array(current_scaled)
# mse = torch.mean((predicted_current - current_scaled) ** 2).item()
# rmse = np.sqrt(mse)
MSE = mean_squared_error(current, predicted_current_vine_copula)
RMSE_vine_copula = sqrt(MSE)
print(f'RMSE_vine_copula: {RMSE_vine_copula}')



# copula进行预测
with torch.no_grad():
    predicted_current_copula = model_copula(input_features)
predicted_current_copula[predicted_current_copula < 0] = 0
length = len(input_features)
for i in range(length):
    if input_features[i, 1] == 0:
        predicted_current_copula[i] = 0

# 反归一化预测结果
predicted_current_copula = x_scaler.inverse_transform(predicted_current_copula)


# 计算 RMSE
# current_scaled = np.array(current_scaled)
# mse = torch.mean((predicted_current - current_scaled) ** 2).item()
# rmse = np.sqrt(mse)
MSE = mean_squared_error(current, predicted_current_copula)
RMSE_copula = sqrt(MSE)
print(f'RMSE_copula: {RMSE_copula}')


# one_step进行预测
with torch.no_grad():
    predicted_current_one_step = model_one_step(input_features)
predicted_current_one_step[predicted_current_one_step < 0] = 0
length = len(input_features)
for i in range(length):
    if input_features[i, 1] == 0:
        predicted_current_one_step[i] = 0

# 反归一化预测结果
predicted_current_one_step = x_scaler.inverse_transform(predicted_current_one_step)


# 计算 RMSE
# current_scaled = np.array(current_scaled)
# mse = torch.mean((predicted_current - current_scaled) ** 2).item()
# rmse = np.sqrt(mse)
MSE = mean_squared_error(current, predicted_current_one_step)
RMSE_one_step = sqrt(MSE)
print(f'RMSE_one_step: {RMSE_one_step}')

# quantile进行预测
with torch.no_grad():
    predicted_current_quantile = model_quantile(input_features)
predicted_current_quantile[predicted_current_quantile < 0] = 0
length = len(input_features)
for i in range(length):
    if input_features[i, 1] == 0:
        predicted_current_quantile[i] = 0

# 反归一化预测结果
predicted_current_quantile = x_scaler.inverse_transform(predicted_current_quantile)


# 计算 RMSE
# current_scaled = np.array(current_scaled)
# mse = torch.mean((predicted_current - current_scaled) ** 2).item()
# rmse = np.sqrt(mse)
MSE = mean_squared_error(current, predicted_current_quantile)
RMSE_quantile = sqrt(MSE)
print(f'RMSE_quantile: {RMSE_quantile}')

# 未处理数据进行预测
with torch.no_grad():
    predicted_current_original = model_original(input_features)
predicted_current_original[predicted_current_original < 0] = 0
length = len(input_features)
for i in range(length):
    if input_features[i, 1] == 0:
        predicted_current_original[i] = 0

# 反归一化预测结果
predicted_current_original = x_scaler.inverse_transform(predicted_current_original)


# 计算 RMSE
# current_scaled = np.array(current_scaled)
# mse = torch.mean((predicted_current - current_scaled) ** 2).item()
# rmse = np.sqrt(mse)
MSE = mean_squared_error(current, predicted_current_original)
RMSE_original = sqrt(MSE)
print(f'RMSE_original: {RMSE_original}')
# 可视化预测结果
plt.plot(predicted_current_original, 'k', linewidth=1, label='predicted_current_original')
plt.plot(predicted_current_quantile, 'g', linewidth=1, label='predicted_current_quantile')
plt.plot(predicted_current_vine_copula, 'r', linewidth=1, label='predicted_current_vine_copula')
plt.plot(predicted_current_copula, 'c', linewidth=1, label='predicted_current_copula')
plt.plot(predicted_current_one_step, 'm', linewidth=1, label='predicted_current_one_step')
plt.plot(current, 'b', linewidth=1, label='current_scaled ')
plt.legend()
plt.show()
scipy.io.savemat('predicted_current_original.mat', {'predicted_current_original': predicted_current_original})
scipy.io.savemat('predicted_current_vine_copula.mat', {'predicted_current_vine_copula': predicted_current_vine_copula})
scipy.io.savemat('predicted_current_copula.mat', {'predicted_current_copula': predicted_current_copula})
scipy.io.savemat('predicted_current_one_step.mat', {'predicted_current_one_step': predicted_current_one_step})
scipy.io.savemat('predicted_current_quantile.mat', {'predicted_current_quantile': predicted_current_quantile})
scipy.io.savemat('current.mat', {'current': current})