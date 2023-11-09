'''
这个判断压井液密度的方法是
用区块中井数大于等于3的区块
来预测
与初始版本的区别是，本次预测的指标就是mean、max、min
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
        return {
            'mse': mean_squared_error(Y_pred, Y_test)
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

    def train(self):
        save = SaveToCsv(r'E:\项目\Intelligent_well_control\reports\deal_density_report.csv')

        block_st = self.find()
        for block_id in block_st.keys():
            print(block_id)
            cur_data = self.data[self.data['block_id'] == block_id]

            # 只取溢流状态下的数据
            cur_data = cur_data[cur_data['overflow_detected'] == 1]
            well_id_list = cur_data['well_id'].unique().tolist()

            print(well_id_list)

            labels = 'deal_density'
            rem_col_list = ['id', 'well_id', 'time', 'overflow_flag',
                            'work_state', 'invader_type', 'kill_main_method_x',
                            'deal_density', 'overflow_detected', 'block_id',
                            'standpipe_pressure', 'casing_pressure']
            feature_names = list(
                filter(lambda x: x not in rem_col_list, cur_data.columns))

            test_well_ids = random.sample(well_id_list, len(well_id_list) * 4 // 10)
            train_well_ids = [well_id for well_id in well_id_list if well_id not in test_well_ids]

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

            details_score = self.pred(model, test)
            print(details_score)

            save.save({
                '任务名': '压井液密度预测',
                '时间': datetime.datetime.now(),
                '区块号': block_id,
                '训练井号': train_well_ids,
                '训练数据量': X_train.shape[0],
                '测试井号': test_well_ids,
                '测试数据量': X_test.shape[0],
                'mse': score['mse'],
                '其他': details_score,
            })

        # 留个空行，方便区分每次实验
        save.save(
            {'任务名': '', '时间': '', '区块号': '', '训练井号': '', '训练数据量': '', '测试井号': '', '测试数据量': '',
             '准确率': '', '其他': ''})


if __name__ == '__main__':
    path = r'E:\data\压井\新数据\间接数据\大区块数据.csv'
    CheckDealDensity(pd.read_csv(path)).train()
