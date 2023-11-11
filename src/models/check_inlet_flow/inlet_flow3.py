'''
这个判断inlet_flow的方法是
用区块中井数大于等于3的区块
来预测
与版本3的区别是：
把每个井都做为测试井试一下，看看哪些好，哪些差！
这个作为ppt的实验
'''
import random
import time
import numpy as np
import datetime
import pandas as pd
from Intelligent_well_control.src.models.utils.save_to_csv import SaveToCsv
from Intelligent_well_control.src.models.LGBM import LGBModel
from sklearn.metrics import mean_squared_error


class CheckDealDensity():
    def __init__(self, data=None):
        self.data = data

    # 找到区号中大于等于3口井的block_id
    # 返回类型是字典{2:[1,2,5], 5:[6, 7, 8]}
    def find(self, cnt=3):
        block_ids = self.data['block_id'].unique().tolist()

        st = {}
        for block_id in block_ids:
            well_ids = self.data[self.data['block_id'] == block_id]['well_id'].unique().tolist()
            if len(well_ids) >= cnt:
                st[block_id] = well_ids

        return st

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

    def pred(self, model, test):
        res = {}
        for well_id in test.keys():
            cur_res = {}
            X_test, Y_test = test[well_id]
            Y_pred = model.other_pred(X_test)
            min_value = np.min(Y_pred)
            max_value = np.max(Y_pred)
            mean_value = np.min(Y_pred)
            cur_res['max'] = round(max_value, 2)
            cur_res['min'] = round(min_value, 2)
            cur_res['mean'] = round(mean_value, 2)
            cur_res['true'] = np.min(Y_test)
            res[well_id] = cur_res

        return res

    def train(self, block_id):
        save = SaveToCsv(r'E:\项目\Intelligent_well_control\reports\inlet_flow_3_report.csv')
        cur_data = self.data[self.data['block_id'] == block_id]
        # cur_data = cur_data[cur_data['overflow_detected'] == 1]
        cur_data.loc[cur_data['inlet_flow'] <= 0, 'inlet_flow'] = 0
        # cur_data.loc[cur_data['inlet_flow'] == -1, 'inlet_flow'] = 0
        well_id_list = cur_data['well_id'].unique().tolist()

        for test_well_id in well_id_list:
            test_well_ids = [test_well_id]
            train_well_ids = [well_id for well_id in well_id_list if well_id not in test_well_ids]

            labels = 'inlet_flow'
            rem_col_list = ['id', 'well_id', 'time', 'overflow_flag',
                            'work_state', 'invader_type', 'kill_main_method_x',
                            'deal_density', 'overflow_detected', 'block_id',
                            'standpipe_pressure', 'casing_pressure', 'inlet_flow']
            feature_names = list(
                filter(lambda x: x not in rem_col_list, cur_data.columns))

            X_train = cur_data[cur_data['well_id'].isin(train_well_ids)][feature_names]
            Y_train = cur_data[cur_data['well_id'].isin(train_well_ids)][labels]

            # 这里注意一下区别，每一个预测的井号单独进行预测，搞一个字典记录吧

            test = {}
            for test_id in test_well_ids:
                test[test_id] = (cur_data[cur_data['well_id'] == test_id][feature_names],
                                 cur_data[cur_data['well_id'] == test_id][labels])

            X_test = cur_data[cur_data['well_id'].isin(test_well_ids)][feature_names]
            Y_test = cur_data[cur_data['well_id'].isin(test_well_ids)][labels]

            # print(Y_train['overflow_detected'].value_counts())

            print(X_train.shape, X_test.shape, Y_train.shape)

            model = LGBModel(X_train=X_train, Y_train=Y_train, X_test=X_test)

            Y_pred = model.self_pred()

            score = self.metrics(Y_test, Y_pred)

            save.save({
                '任务名': '入口排量模型3',
                '时间': datetime.datetime.now(),
                '训练数据量': X_train.shape[0],
                '测试数据量': X_test.shape[0],
                'mse': score['mse'],
                '区块号': block_id,
                # '训练井号': train_well_ids,
                '测试井号': test_well_ids,
                'mean_pred': score['mean_pred'],
                'std_pred': score['std_pred'],
                'median_pred': score['median_pred'],
                'mean_true': score['mean_true'],
                'std_true': score['std_true'],
                'median_true': score['median_true'],
                '其他': ' ',
            })

        save.save(
            {'任务名': '',
             '时间': '',
             '训练数据量': '',
             '测试数据量': '',
             'mse': '',
             '区块号': '区块号',
             # '训练井号': '训练井号',
             '测试井号': '测试井号',
             'mean_pred': 'mean_pred',
             'std_pred': 'std_pred',
             'median_pred': 'median_pred',
             'mean_true': 'mean_true',
             'std_true': 'std_true',
             'median_true': 'median_true',
             }
        )




if __name__ == '__main__':
    path = r'E:\data\压井\新数据\间接数据\大区块数据.csv'
    checkDealDensity = CheckDealDensity(pd.read_csv(path))
    for block_id in checkDealDensity.find():
        checkDealDensity.train(block_id)
