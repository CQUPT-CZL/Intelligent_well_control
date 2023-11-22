# 使用所有的井的数据跑一边kill_main_method，每一口井都单独进行预测。

import random
import time
import datetime
import numpy as np
from datetime import datetime
import pandas as pd
import lightgbm as lgb
from Intelligent_well_control.src.models.utils.save_to_csv import SaveToCsv
from Intelligent_well_control.src.models.LGBM import LGBModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error


class CheckKillMainMethod1():
    def __init__(self, data_file = None, save_file = None,  is_save = False):
        self.data_file = data_file
        self.save_file = save_file
        self.is_save = is_save
        if is_save:
            self.save = SaveToCsv(save_file)

   # 分类问题的指标
    def metrics(self, Y_test, Y_pred):
        # print(Y_test, Y_pred)
        unique_elements, counts = np.unique(Y_pred, return_counts=True)
        max_count_index = np.argmax(counts)
        mode = unique_elements[max_count_index]
        return {
            'accuracy_score': accuracy_score(Y_pred, Y_test),
            'mode': mode
        }

    def train(self):

        data = pd.read_csv(self.data_file)
        # data = data[data['overflow_detected'] == 1]

        #输出所有井号
        well_id_lists = data['well_id'].unique().tolist()
        print(len(well_id_lists), well_id_lists)

        labels = 'kill_main_method'
        rem_col_list = ['id', 'well_id', 'time', 'overflow_flag',
                        'work_state', 'invader_type',
                        'deal_density', 'overflow_detected', 'block_id',
                        'standpipe_pressure', 'casing_pressure']

        feature_names = list(
            filter(lambda x: x not in rem_col_list, data.columns))
        print(feature_names)
        print(data.columns)

        for test_well_id in well_id_lists:
            test_well_ids = [test_well_id]
            train_well_ids = [well_id for well_id in well_id_lists if well_id not in test_well_ids]


            print(test_well_ids)

            # X_train =
            # Y_train =
            # X_test =
            Y_test = data[data['well_id'].isin(test_well_ids)][labels]
            # print(Y_test)

            model = LGBModel(type = 'classifier',
                X_train=data[data['well_id'].isin(train_well_ids)][feature_names],
                Y_train=data[data['well_id'].isin(train_well_ids)][labels],
                X_test=data[data['well_id'].isin(test_well_ids)][feature_names])

            Y_pred = model.self_pred()
            print(Y_pred)

            score = self.metrics(Y_test, Y_pred)
            if self.is_save:
                print(self.save_file)
                self.save.save({
                    'work': 'kill_main_method1',
                    # '时间': datetime.now().date(),
                    # '区块号': 'block_id',
                    # '训练井号': train_well_ids,
                    # '训练数据量': X_train.shape[0],
                    'test_well_id': test_well_ids,
                    # '测试数据量': X_test.shape[0],
                    'accuracy': score['accuracy_score'],
                    'true': Y_test.iloc[0],
                    'pred': score['mode'],
                    'other': ''
                })

        # 留个空行，方便区分每次实验
        # save.save({'任务名': '', '时间': '', '区块号': '', '训练井号': '', '训练数据量': '', '测试井号': '', '测试数据量': '', '准确率': '', '其他': ''})


if __name__ == '__main__':

    is_remote = True
    is_save = True

    data_file = r'E:\data\压井\新数据\间接数据\一半总数据2.csv'
    save_file = r'E:\项目\Intelligent_well_control\reports\kill_main_method\kill_main_method_1_report.csv'
    if is_remote:
        data_file = r'~/data/压井/新数据/间接数据/一半总数据2.csv'
        save_file = r'/home/czl/project/Intelligent_well_control/reports/kill_main_method/kill_main_method_1_report.csv'

    print(data_file, save_file)

    CheckKillMainMethod1(data_file = data_file, save_file = save_file, is_save = is_save).train()
