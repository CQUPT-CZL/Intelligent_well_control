import lightgbm as lgb

'''
模块说明
本模块作为一个可以泛化的模型
读入的数据是X_train, Y_train, X_test
返回的数据是Y_pred
'''


class LGBModel():
    def __init__(self, type = 'regressor', X_train = None, Y_train = None, X_test = None):
        self.model = lgb.LGBMRegressor(n_estimators=4000)
        if type != 'regressor':
            self.model = lgb.LGBMClassifier(n_estimators=4000)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

    def train_and_pred(self):
        self.model.fit(self.X_train, self.Y_train)
        return self.model.predict(self.X_test)

