# 消融实验
# 取出10口井后，先把50口井的数据依次训练预测（完成后用这个模型预测那10口井）
# 找到最差的3三口井，然后下一轮中删除掉
# 再用47口井去训练，依次操作5轮左右。对比那10口井的结果！

import random
import numpy as np
import time
import datetime
import pandas as pd
from Intelligent_well_control.src.models.utils.save_to_csv import SaveToCsv
from Intelligent_well_control.src.models.utils.plt import PLT
from Intelligent_well_control.src.models.LGBM import Model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score

# 定义一些变量

is_local = False
data_file = r'E:\data\压井\新数据\间接数据\一半总数据2.csv'
save_file = r'E:\项目\Intelligent_well_control\reports\overflow_4.4_report.csv'

if not is_local:
    data_file = r'~/data/压井/新数据/间接数据/一半总数据2.csv'
    save_file = '/home/czl/project/Intelligent_well_control/reports/check_overflow/overflow_4.6_report.csv'

epoch_remove_cnt = 3
epoch = 8
model_name = 'xgb'
# 读入数据
data = pd.read_csv(data_file)

# 获取特征列
labels = 'overflow_detected'
rem_col_list = ['id', 'well_id', 'time', 'overflow_flag',
                'work_state', 'invader_type', 'kill_main_method_x',
                'deal_density', 'overflow_detected', 'block_id',
                'standpipe_pressure', 'casing_pressure']
feature_names = list(
    filter(lambda x: x not in rem_col_list, data.columns))

# 获取井号
all_well_ids = data['well_id'].unique().tolist()

# 随机选出10口井
# verify_well_ids = random.sample(all_well_ids, 10)
verify_well_ids = [111, 5, 55, 71, 100, 19, 90, 65, 50, 13]

res = {}
for verify_well_id in verify_well_ids:
    res[verify_well_id] = []

# 用于训练模型的井
cur_well_ids = [well_id for well_id in all_well_ids if well_id not in verify_well_ids]


save = SaveToCsv(save_file)

# 开始消融实验
for epo in range(epoch):
    print(epo, 'ext')
    st = []
    for test_well_id in cur_well_ids:
        test_well_ids = [test_well_id]
        train_well_ids = [well_id for well_id in cur_well_ids if well_id not in test_well_ids]

        # X_train =
        # Y_train =
        X_test = data[data['well_id'].isin(test_well_ids)][feature_names]
        Y_test = data[data['well_id'].isin(test_well_ids)][labels]

        model = Model(model_name=model_name,
            type_name='classifier',
                         X_train=data[data['well_id'].isin(train_well_ids)][feature_names],
                         Y_train=data[data['well_id'].isin(train_well_ids)][labels])
        Y_pred = model.pred(X_test)

        # 画个图
        # plt = PLT(1)
        # plt.show2(Y_pred = Y_pred, Y_true = Y_test)


        # 获取准确率
        acc = accuracy_score(Y_pred, Y_test)
        st.append((test_well_ids, acc))
        print(test_well_ids, acc)

    # st数组存储对实验结果,对他按照acc升序排序
    st.sort(key = lambda x : x[1])
    print(st[ : 10])
    # 从井中剔除掉这些acc差的值，我们认为这个acc差的就是井数据噪声大的
    rem_well_ids = []
    for i in range(epoch_remove_cnt):
        print(st[i][0][0], ' had removed')
        rem_well_ids.append(st[i][0][0])
        cur_well_ids.remove(st[i][0][0])


    # 对那十口井进行验证
    # X_train =
    # Y_train =

    model = Model(model_name=model_name,
        type_name='classifier',
                     X_train=data[data['well_id'].isin(cur_well_ids)][feature_names],
                     Y_train=data[data['well_id'].isin(cur_well_ids)][labels])

    tem_save = {'epoch' : epo + 1}
    for test_id in verify_well_ids:
        # print(test_id, '----verify')
        X_test = data[data['well_id'] == test_id][feature_names]
        Y_test = data[data['well_id'] == test_id][labels]

        Y_pred = model.pred(X_test)

        acc = accuracy_score(Y_pred, Y_test)

        tem_save[test_id] = "{:.1%}".format(acc)

        res[test_id].append(np.round(acc, 3))

    tem_save['other'] = f'removed {rem_well_ids}, {model_name}'
    save.save(tem_save)

print(res)

# # 画图
# plt = PLT(res, y_label='acc', x_label='epoch', xticks=list(range(1, 11)),
#           save_file=r'E:\项目\Intelligent_well_control\reports\images\overflow4.png')
# plt.show()











