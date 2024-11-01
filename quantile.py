from math import sqrt

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import  mean_absolute_error
import numpy as np

# 用pandas加载Excel文件
data = pd.read_excel('data_quntile.xlsx', header=None)

# 提取温度、辐照度和电流列的数据
temperature = data.iloc[:, 2]
radiation = data.iloc[:, 3]
current = data.iloc[:, 1]

# 创建MinMaxScaler的实例，用于归一化温度和辐照度

x_scaler = MinMaxScaler()
temperature_scaled = x_scaler.fit_transform(temperature.to_numpy().reshape(-1, 1))
radiation_scaled = x_scaler.fit_transform(radiation.to_numpy().reshape(-1, 1))
#current = current.to_numpy().reshape(-1, 1)
y_scaler = MinMaxScaler()
current_scaled = y_scaler.fit_transform(current.to_numpy().reshape(-1, 1))
# 将归一化后的温度和辐照度合并成输入特征
input_features = torch.tensor(list(zip(temperature_scaled, radiation_scaled)), dtype=torch.float32)

# 将电流作为输出特征
output_feature = torch.tensor(current_scaled, dtype=torch.float32)

# 划分训练集和测试集
#x_train, x_test, y_train, y_test = train_test_split(input_features, output_feature, test_size=0.2, random_state=42)
split_index = int(0.8 * len(input_features))
x_train, x_test = input_features[:split_index], input_features[split_index:]
y_train, y_test = output_feature[:split_index], output_feature[split_index:]
# 定义Bi-LSTM模型
class BiLSTMNet(nn.Module):
    def __init__(self, input_size):
        super(BiLSTMNet, self).__init__()
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
model = BiLSTMNet(input_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

# 测试模型
with torch.no_grad():
    test_output = model(x_test)
    test_loss = criterion(test_output, y_test)
    print(f'Test Loss: {test_loss.item()}')

# 保存模型参数到文件
torch.save(model.state_dict(), 'module/model_parameters_quntile.pth')
test_output = y_scaler.inverse_transform(test_output.numpy())
y_test = y_scaler.inverse_transform(y_test.numpy())

MSE = mean_squared_error(y_test, test_output)
RMSE = sqrt(MSE)
#R2 = r2_score(y_test, test_output)
#MAE = mean_absolute_error(y_test, test_output)
# MAPE = method.MAPE_value(series_y, pred)
#print(torch.name, ' :')
print(' MSE: {:.3f}'.format(MSE))
print(' RMSE: {:.3f}'.format(RMSE))
#print(' MAE: {:.3f}'.format(MAE))
#print(' R2: {:.3f}'.format(R2))

plt.plot(test_output, 'r', label='test_output')
plt.plot(y_test, 'b', label='y_test')
plt.legend()
plt.show()

