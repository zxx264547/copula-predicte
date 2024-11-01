import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler
# 用pandas加载Excel文件
data = pd.read_excel('data.xlsx')
data.columns = ["V", "I", "T", "E"]
scaler = MinMaxScaler()
data = scaler.fit_transform(data.to_numpy())


# 提取特征和目标变量
def create_dataset(data, target_features, input_features):
    data_x = data[target_features]
    data_y = data[input_features]
    data_x = torch.from_numpy(data_x).float()

    data_x = data_x.reshape(data_x.shape[0], 1, data_x.shape[1])
    data_y = torch.squeeze(torch.from_numpy(data_y).float())
    return data_x, data_y
data_x,data_y = create_dataset(data,[3,4],[2])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=100, output_size=1, num_layers=1):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x) # 未在不同序列中传递hidden_state
        return self.reg(x)


rnn = RNN(input_size=2)
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
loss_func = nn.MSELoss()
epochs = 100
print(rnn)

for e in range(epochs):
    # 前向传播
    y_pred = rnn(x_train)
    y_pred = torch.squeeze(y_pred)

    loss = loss_func(y_pred, y_train)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 20 == 0:
        print('Epoch:{}, Loss:{:.5f}'.format(e, loss.item()))
#y_pred = scaler.inverse_transform(data_standardized)
plt.plot(y_pred.detach().numpy(), 'r', label='y_pred')
plt.plot(y_train.detach().numpy(), 'b', label='y_train')
plt.legend()
plt.show()
