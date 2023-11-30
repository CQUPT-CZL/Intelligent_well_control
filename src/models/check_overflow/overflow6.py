# 用所有的井的数据跑, 每一口井都测试一边吧
# 模型3的进阶版
# 尝试不是一次性入读内存数据
import random
import time
import datetime
from datetime import datetime
import pandas as pd
import lightgbm as lgb
from Intelligent_well_control.src.models.utils.save_to_csv import SaveToCsv
from Intelligent_well_control.src.models.LGBM import LGBModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score


class CheckOverflow06():
    def __init__(self, data_file=None, save_file=None, is_save=False):
        self.data_file = data_file
        self.save_file = save_file
        self.is_save = is_save
        if is_save:
            self.save = SaveToCsv(save_file)

    def metrics(self, Y_test, Y_pred):
        # print(Y_test, Y_pred)
        return {
            'accuracy_score': accuracy_score(Y_pred, Y_test)
        }


    def csv_generator(self, filename, chunksize):
        for chunk in pd.read_csv(filename, chunksize=chunksize):
            yield chunk


    def train(self):
        # save =
        # save = SaveToCsv(r'E:\项目\Intelligent_well_control\reports\overflow_3_report.csv')
        # save = SaveToCsv('/home/czl/project/Intelligent_well_control/reports/overflow_6_report.csv')
        # save.save({})
        well_id_list = pd.read_csv(self.data_file, usecols=['well_id'])['well_id'].unique().tolist()
        all_columns_list = pd.read_csv(self.data_file, nrows=2).columns.tolist()

        print(len(well_id_list), well_id_list)

        labels = 'overflow_detected'
        rem_col_list = ['id', 'well_id', 'time', 'overflow_flag',
                        'work_state', 'invader_type', 'kill_main_method_x',
                        'deal_density', 'overflow_detected', 'block_id',
                        'standpipe_pressure', 'casing_pressure']
        feature_names = list(
            filter(lambda x: x not in rem_col_list, all_columns_list))

        for test_well_id in well_id_list:

            # if test_well_id != 60:
            #     continue

            test_well_ids = [test_well_id]

            train_well_ids = [well_id for well_id in well_id_list if well_id not in test_well_ids]

            print(test_well_ids)

            generator = self.csv_generator(self.data_file, 50000)

            model = lgb.LGBMClassifier(n_estimators=500)

            X_test, Y_test = None, None

            for data in generator:
                # print(data.shape)
                X_train = data[data['well_id'].isin(train_well_ids)][feature_names]
                Y_train = data[data['well_id'].isin(train_well_ids)][labels]
                sub_X_test = data[data['well_id'].isin(test_well_ids)][feature_names]
                sub_Y_test = data[data['well_id'].isin(test_well_ids)][labels]

                # 将测试数据拼起来
                if sub_X_test.shape[0] > 0:
                    if X_test is None:
                        X_test = sub_X_test
                        Y_test = sub_Y_test
                    else:
                        X_test = pd.concat([X_test, sub_X_test])
                        Y_test = pd.concat([Y_test, sub_Y_test])
                    print("---------------")
                    print(X_test.shape)

                if X_train.shape[0] > 0:
                    print(X_train.shape, Y_train.shape)
                    model.fit(X_train, Y_train)
            # print(Y_test)

            print('-----X_test----')
            # print(X_test, Y_test)
            print(X_test.shape, Y_test.shape)
            Y_pred = model.predict(X_test)
            print(Y_pred)


            score = self.metrics(Y_test, Y_pred)

            print(score['accuracy_score'])
            if self.is_save:
                self.save.save({
                    'work': 'is_overflow',
                    # '时间': datetime.now().date(),
                    # '区块号': 'block_id',
                    # '训练井号': train_well_ids,
                    # '训练数据量': X_train.shape[0],
                    'test_well_id': test_well_ids,
                    # '测试数据量': X_test.shape[0],
                    'accuracy': score['accuracy_score'],
                    'other': 'y_is_detected'
                })
                # 留个空行，方便区分每次实验
                # save.save({'任务名': '', '时间': '', '区块号': '', '训练井号': '', '训练数据量': '', '测试井号': '', '测试数据量': '', '准确率': '', '其他': ''})


if __name__ == '__main__':
    is_remote = False
    is_save = True

    data_file = r'E:\data\压井\新数据\间接数据\一半总数据2.csv'
    save_file = r'E:\项目\Intelligent_well_control\reports\check_overflow\overflow_6_report.csv'
    if is_remote:
        data_file = r'~/data/压井/新数据/间接数据/一半总数据2.csv'
        save_file = r'/home/czl/project/Intelligent_well_control/reports/check_overflow/overflow_6_report.csv'

    print(data_file, save_file)

    CheckOverflow06(data_file=data_file, save_file=save_file, is_save=is_save).train()
