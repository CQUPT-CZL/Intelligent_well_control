# 使用所有的井的数据跑一边inlet_flow，每一口井都单独进行预测。
#
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


class CheckInletFlow4():
    def __init__(self, data_file = None, save_file = None,  is_save = False):
        self.data_file = data_file
        self.save_file = save_file
        self.is_save = is_save
        if is_save:
            self.save = SaveToCsv(save_file)

   # 回归问题的指标
    def metrics(self, Y_test, Y_pred):
        # print(Y_test, Y_pred)
        max_pred = np.max(Y_pred)
        min_pred = np.min(Y_pred)
        mean_pred = np.mean(Y_pred)
        std_pred = np.std(Y_pred)
        median_pred = np.median(Y_pred)

        max_true = np.max(Y_test)
        min_true = np.min(Y_test)
        mean_true = np.mean(Y_test)
        std_true = np.std(Y_test)
        median_true = np.median(Y_test)

        return {
            'mse': mean_squared_error(Y_pred, Y_test),
            'max_pred': np.round(max_pred, 2),
            'min_pred': np.round(min_pred, 2),
            'mean_pred': np.round(mean_pred, 2),
            'std_pred': np.round(std_pred, 2),
            'median_pred': np.round(median_pred, 2),
            'max_true': np.round(max_true, 2),
            'min_true': np.round(min_true, 2),
            'mean_true': np.round(mean_true, 2),
            'std_true': np.round(std_true, 2),
            'median_true': np.round(median_true, 2),
        }

    def train(self):

        data = pd.read_csv(self.data_file)
        data = data[data['overflow_detected'] == 1]

        #输出所有井号
        well_id_lists = data['well_id'].unique().tolist()
        print(len(well_id_lists), well_id_lists)

        labels = 'inlet_flow'
        rem_col_list = ['id', 'well_id', 'time', 'overflow_flag',
                        'work_state', 'invader_type', 'kill_main_method_x',
                        'deal_density', 'overflow_detected', 'block_id',
                        'standpipe_pressure', 'casing_pressure', 'inlet_flow']
        feature_names = list(
            filter(lambda x: x not in rem_col_list, data.columns))

        for test_well_id in well_id_lists:
            test_well_ids = [test_well_id]
            train_well_ids = [well_id for well_id in well_id_lists if well_id not in test_well_ids]


            print(test_well_ids)

            # X_train =
            # Y_train =
            # X_test =
            Y_test = data[data['well_id'].isin(test_well_ids)][labels]
            # print(Y_test)

            model = LGBModel( X_train=data[data['well_id'].isin(train_well_ids)][feature_names],
                             Y_train=data[data['well_id'].isin(train_well_ids)][labels],
                             X_test=data[data['well_id'].isin(test_well_ids)][feature_names])

            Y_pred = model.self_pred()
            print(Y_pred)

            score = self.metrics(Y_test, Y_pred)
            if self.is_save:
                print(self.save_file)
                self.save.save({
                    'test_well_id': test_well_ids,
                    'work': 'inlet_flow4',
                    # '时间': datetime.datetime.now(),
                    # '训练数据量': X_train.shape[0],
                    # '测试数据量': X_test.shape[0],
                    'mse': score['mse'],
                    # '区块号': block_id,
                    # '训练井号': train_well_ids,
                    'mean_pred': score['mean_pred'],
                    'std_pred': score['std_pred'],
                    'median_pred': score['median_pred'],
                    'min_pred': score['min_pred'],
                    'max_pred': score['max_pred'],
                    'mean_true': score['mean_true'],
                    'std_true': score['std_true'],
                    'median_true': score['median_true'],
                    'min_true': score['min_true'],
                    'max_true': score['max_true'],
                    'mean_diff': np.abs(score['mean_pred'] - score['mean_true']),
                    'other': ' ',
                })

        # 留个空行，方便区分每次实验
        # save.save({'任务名': '', '时间': '', '区块号': '', '训练井号': '', '训练数据量': '', '测试井号': '', '测试数据量': '', '准确率': '', '其他': ''})


if __name__ == '__main__':

    is_remote = True
    is_save = True

    data_file = r'E:\data\压井\新数据\间接数据\一半总数据2.csv'
    save_file = r'E:\项目\Intelligent_well_control\reports\inlet_flow\inlet_flow_4_report.csv'
    if is_remote:
        data_file = r'~/data/压井/新数据/间接数据/一半总数据2.csv'
        save_file = r'/home/czl/project/Intelligent_well_control/reports/inlet_flow/inlet_flow_4_report.csv'

    print(data_file, save_file)

    CheckInletFlow4(data_file = data_file, save_file = save_file, is_save = is_save).train()
