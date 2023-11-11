'''
与模型1的区别在于
他预测每一个区块的每一口井的结果
'''
import random
import time
import datetime
import pandas as pd
from Intelligent_well_control.src.models.utils.save_to_csv import SaveToCsv
from Intelligent_well_control.src.models.LGBM import LGBModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score

class CheckOverflow01():
    def __init__(self, data = None):
        self.data = data


    # 找到区号中大于等于3口井的block_id
    # 返回类型是字典{2:[1,2,5], 5:[6, 7, 8]}
    def find(self, cnt = 3):
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
            'accuracy_score' : accuracy_score(Y_pred, Y_test)
        }


    def train(self):
        save = SaveToCsv(r'E:\项目\Intelligent_well_control\reports\overflow_2_report.csv')

        block_st = self.find()
        for block_id in block_st.keys():
            cur_data = self.data[self.data['block_id'] == block_id]
            well_id_list = cur_data['well_id'].unique().tolist()
            for test_well_id in well_id_list:
                labels = 'overflow_detected'
                rem_col_list = ['id', 'well_id', 'time', 'overflow_flag',
                                'work_state', 'invader_type', 'kill_main_method_x',
                                'deal_density', 'overflow_detected', 'block_id',
                                'standpipe_pressure', 'casing_pressure']
                feature_names = list(
                    filter(lambda x: x not in rem_col_list, cur_data.columns))

                test_well_ids = [test_well_id]
                train_well_ids = [well_id for well_id in well_id_list if well_id not in test_well_ids]

                X_train = cur_data[cur_data['well_id'].isin(train_well_ids)][feature_names]
                Y_train = cur_data[cur_data['well_id'].isin(train_well_ids)][labels]
                X_test = cur_data[cur_data['well_id'].isin(test_well_ids)][feature_names]
                Y_test = cur_data[cur_data['well_id'].isin(test_well_ids)][labels]

                # 标准化数据
                # scaler = StandardScaler(```````````)
                # X_train = scaler.fit_transform(X_train)
                # X_test = scaler.fit_transform(X_test)```````````

                # print(Y_train['overflow_detected'].value_counts())

                print(X_train.shape, X_test.shape, Y_train.shape)

                model = LGBModel(type = 'classifier', X_train = X_train, Y_train = Y_train, X_test = X_test)
                Y_pred = model.self_pred()

                score = self.metrics(Y_test, Y_pred)

                save.save({
                    '任务名': '预测是否溢流',
                    '时间' : datetime.datetime.now().date(),
                    '区块号' : block_id,
                    # '训练井号' : train_well_ids,
                    '训练数据量' : X_train.shape[0],
                    '测试井号' : test_well_ids, #其实就一个元素，不用列表了！
                    '测试数据量' : X_test.shape[0],
                    '准确率' : score['accuracy_score'],
                    '其他' : 'y是detected'
                })

            # 留个空行，方便区分每次实验
            save.save({'任务名': '',
                       '时间': '',
                       '区块号': '区块号',
                       # '训练井号': '训练井号',
                       '训练数据量': '训练数据量',
                       '测试井号': '测试井号',
                       '测试数据量': '测试数据量',
                       '准确率': '准确率',
                       '其他': ''})


if __name__ == '__main__':
    path = r'E:\data\压井\新数据\间接数据\大区块数据.csv'

    CheckOverflow01(pd.read_csv(path)).train()





