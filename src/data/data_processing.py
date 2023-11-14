'''
数据处理
将四个小表的数据与大表数据融合
小表数据中若有的井有多行，则依次堆叠
完成后的数据只有int或者float类型
'''


import pandas as pd
import numpy as np
import re


class DataProcessing():

    # 取得well_log_data表
    def get_well_log_data(self, path='E:\data\压井\新数据\原始数据'):
        all_data_list = []
        for i in range(1, 18):
            str_num = str(i)

            if i < 10:
                str_num = '0' + str_num
            print(str_num)
            all_data_list.append(pd.read_csv(path + '\录井仪' + str_num + '.csv'))
        all_data = pd.concat(all_data_list)
        return all_data

    # 取得well_base_info表
    def get_well_base_info_data(self, path='E:\data\压井\新数据\原始数据'):
        well_base_info_data = pd.read_csv(path + '\井基础信息.csv')
        return well_base_info_data

    # 取得well_body表
    def get_well_body_data(self, path='E:\data\压井\新数据\原始数据'):
        well_body_data = pd.read_csv(path + '\井身数据.csv')
        return well_body_data

    # 取得well_geology表
    def get_well_geology_data(self, path='E:\data\压井\新数据\原始数据'):
        well_geology_data = pd.read_csv(path + '\地质数据.csv')
        return well_geology_data

    # 取得well_overflow_info表
    def get_well_overflow_info_data(self, path='E:\data\压井\新数据\原始数据'):
        well_overflow_info_data = pd.read_csv(path + '\溢流信息.csv')
        return well_overflow_info_data

    # 将多行数据处理为一行
    def duo_to_one(self, df, id):
        # 创建示例DataFrame

        # 添加多级列名

        df['variable'] = df.groupby(id).cumcount() + 1
        columns = df.columns.tolist()
        columns.remove(id)

        wide_df = df.pivot(index=id, columns='variable', values=columns)

        # 重置列索引
        wide_df.reset_index(inplace=True)

        # 重命名列名
        wide_df.columns = [f'{col[0]}{col[1]}' if col[1] else col[0] for col in wide_df.columns]
        return wide_df

    def save(self, data, path=r'E:\data\压井\新数据\间接数据'):
        data.to_csv(path + '\一半总数据2.csv', index=False)

    def process(self, is_debug=False):

        print('加载五个数据表')
        if is_debug:
            well_log_data = None
        else:
            well_log_data = self.get_well_log_data()
        well_base_info_data = self.get_well_base_info_data()
        well_body_data = self.get_well_body_data()
        well_geology_data = self.get_well_geology_data()
        well_overflow_info_data = self.get_well_overflow_info_data()


        print('处理质地信息表')
        well_geology_data = well_geology_data.sort_values(by = ['well_id', 'sort'])
        well_geology_data.drop(['rock_character', 'complex_layer_tips', 'id', 'sort', 'series', 'group', 'section'], axis=1, inplace=True)
        well_geology_data = self.duo_to_one(well_geology_data, 'well_id')
        print(well_geology_data.shape)

        print('处理井身信息')
        well_body_data = well_body_data.sort_values(by=['well_id', 'drill_sequence'])
        well_body_data.drop(['id', 'event_id', 'casing_pipe_type', 'graded_cementing', 'drill_sequence', 'series', 'group', 'section'], axis=1, inplace=True)
        well_body_data = self.duo_to_one(well_body_data, 'well_id')
        print(well_body_data.shape)

        print('处理溢流信息')
        well_overflow_info_data.drop(['id', 'detected_time', 'deal_end_time', 'shut_in_time',
                                      'kill_start_time', 'direct_reason', 'indirect_reason', 'deal_description',
                                      'detected_staff', 'detected_unit', 'well_control_group', 'plan_design_staff',
                                      'scene_commander', 'assistants', 'scene_commander_level'], axis=1, inplace=True)
        print(well_overflow_info_data.shape)

        print('处理井基础信息')
        well_base_info_data.drop(['oilfield_enterprise_id', 'construct_unit', 'block_specific',
                                  'drilling_enterprise', 'construct_team'], axis=1, inplace=True)
        print(well_base_info_data.shape)

        print('四个子表合并成一个表')
        new_df = pd.merge(well_base_info_data, well_body_data, on='well_id', how='outer')
        new_df = pd.merge(new_df, well_geology_data, on='well_id', how='outer')
        new_df = pd.merge(new_df, well_overflow_info_data, on='well_id', how='outer')
        # new_df = pd.get_dummies(new_df, columns=['block_specific'], dummy_na=True)

        print('处理str特征')
        # str_fea_cols = (['section' + str(i) + '_x' for i in range(1, 21)] +
        #                 ['group' + str(i) + '_x' for i in range(1, 21)] +
        #                 ['series' + str(i) + '_x' for i in range(1, 21)])
        #
        # for fea in str_fea_cols:
        #     if fea in new_df.columns:
        #         new_df[fea] = new_df[fea].fillna('unknow')
        #         new_df[fea] = new_df[fea].map(
        #             dict(zip(new_df[fea].unique(), range(0, new_df[fea].nunique()))))

        # 找出包含汉字元素的列
        chinese_columns = [col for col in new_df.columns if
                           new_df[col].apply(lambda x: bool(re.search('[\u4e00-\u9fff]', str(x)))).any()]

        for fea in chinese_columns:
            if fea in new_df.columns:
                new_df[fea] = new_df[fea].fillna('unknow')
                new_df[fea] = new_df[fea].map(
                    dict(zip(new_df[fea].unique(), range(0, new_df[fea].nunique()))))

        print('删除缺失过半的特征')
        missing_cols = [c for c in new_df if new_df[c].isna().mean() * 100 > 50]
        print(len(missing_cols))
        new_df.drop(missing_cols + ['code'], axis=1, inplace=True)

        print(new_df.info())
        print(new_df.head(3))

        print('类型转换')
        obj_features = new_df.dtypes[new_df.dtypes != 'int'].index
        new_df[obj_features] = new_df[obj_features].astype(float)

        print('空值填写为0')
        new_df.fillna(0, inplace=True)

        print('将overflow那两个的2转为0')
        well_log_data['overflow_flag'].replace(2, 0, inplace=True)
        well_log_data['overflow_detected'].replace(2, 0, inplace=True)

        print('将well_log_data中的负数替换为0')
        for column in well_log_data.columns:
            # 检查列的数据类型是否为整数或浮点数
            if pd.api.types.is_numeric_dtype(well_log_data[column]):
                # 将负数替换为0
                well_log_data[column] = well_log_data[column].apply(lambda x: max(0, x) if pd.notnull(x) else x)

        # 取一半数据
        well_log_data = well_log_data[:: 2]
        new_df.drop_duplicates(subset=['well_id'], inplace=True)
        end_data = pd.merge(well_log_data, new_df, on='well_id', how='left')



        print(f'数据形状：{end_data.shape}')

        print('保存数据')
        self.save(end_data)

        return end_data


if __name__ == '__main__':
    print('start')
    DataProcessing().process()
