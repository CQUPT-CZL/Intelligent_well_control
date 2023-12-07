import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

'''
模块说明
本模块作为一个可以泛化的模型
读入的数据是X_train, Y_train, X_test
返回的数据是Y_pred
'''

class Model():
    def __init__(self, model_name, type_name, X_train, Y_train):
        self.model_name = model_name
        if model_name == 'xgb':
            if type_name == 'classifier':
                params = {
                    'n_estimators': 500
                }
                self.model = xgb.XGBClassifier(**params)
            if type_name == 'regressor':
                params = {
                    'n_estimators': 500
                }
                self.model = xgb.XGBRegressor(**params)
            self.model.fit(X_train, Y_train)

        if model_name == 'lgb':
            if type_name == 'classifier':
                self.model = lgb.LGBMClassifier(n_estimators=500)
            if type_name == 'regressor':
                self.model = lgb.LGBMRegressor(n_estimators=500)
            self.model.fit(X_train, Y_train)

        if model_name == 'svm':
            if type_name == 'classifier':
                self.model = SVC(kernel='linear', C=1.0, n_jobs=-1)
            if type_name == 'regressor':
                self.model = SVR(kernel='linear', C=1.0)
            self.model.fit(X_train, Y_train)

        if model_name == 'rf':
            if type_name == 'classifier':
                self.model = RandomForestClassifier(n_estimators=500, random_state=42)
            if type_name == 'regressor':
                self.model = RandomForestRegressor(n_estimators=500, random_state=42)
            self.model.fit(X_train, Y_train)

        if model_name == 'mlp':

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 转换为PyTorch张量
            features_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
            labels_tensor = torch.tensor(Y_train.values, dtype=torch.long).to(device)

            # 模型参数
            input_size = X_train.shape[1]  # 输入特征的维度，根据数据列数确定
            hidden_size1 = 128  # 第一个隐藏层的大小
            hidden_size2 = 64  # 第二个隐藏层的大小
            hidden_size3 = 32  # 第三个隐藏层的大小
            hidden_size4 = 8  # 第四个隐藏层的大小
            # output_size = len(Y_train['overflow_detected'].unique())  # 输出层的大小，根据标签类别数确定
            output_size = 2  # 输出层的大小，根据标签类别数确定

            # 创建MLP模型
            self.model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size).to(device)

            # 定义损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(self.model.parameters(), lr=0.03)

            # 创建一个数据集
            dataset = TensorDataset(features_tensor, labels_tensor)

            # 定义数据加载器
            batch_size = 64
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # 训练模型
            num_epochs = 2
            for epoch in range(num_epochs):
                for batch_features, batch_labels in data_loader:
                    batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                    # 前向传播
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)

                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')



    def pred(self, X):
        if self.model_name == 'mlp':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X = torch.tensor(X.values, dtype=torch.float32)
            X = X.to(device)

            Y_pred = self.model(X)
            print(Y_pred)
            Y_pred = torch.argmax(Y_pred, dim=1, keepdim=True)
            # print(Y_pred)
            return Y_pred.cpu().detach().numpy()
        else:
            return self.model.predict(X)


# 定义一个4层MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size4, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.softmax(x)
        return x

