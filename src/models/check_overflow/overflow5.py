# 2号区块的消融实验
# 取出4口井2， 38， 108， 78当作测试井，永远不动
# 找到最差的2三口井，然后下一轮中删除掉
# 依次做3次

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
data_file = r'E:\data\压井\新数据\间接数据\大区块数据2.csv'
save_file = r'E:\项目\Intelligent_well_control\reports\overflow_5_report.csv'

if not is_local:
    data_file = '/home/czl/data/压井/新数据/间接数据/大区块数据2.csv'
    save_file = '/home/czl/project/Intelligent_well_control/reports/check_overflow/overflow_5.3_report.csv'

epoch_remove_cnt = 2
epoch = 3

# 读入数据
data = pd.read_csv(data_file)

# 只取2号区块
data = data[data['block_id'] == 2]


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

# 随机选出4口井
verify_well_ids = [2, 38, 108, 78]

res = {}
for verify_well_id in verify_well_ids:
    res[verify_well_id] = []

save = SaveToCsv(save_file)

# 用于训练模型的井
cur_well_ids = [well_id for well_id in all_well_ids if well_id not in verify_well_ids]

# 验证那十口井，并且返回每一口井的预测结果, 并且存入文件
def verify(train_well_ids: list, epo: int, other):
    print(train_well_ids)
    model = Model(model_name = 'mlp',
        type_name='classifier',
                     X_train=data[data['well_id'].isin(train_well_ids)][feature_names],
                     Y_train=data[data['well_id'].isin(train_well_ids)][labels],
                     )

    tem_save = {'epoch': epo + 1}
    for test_well_id in verify_well_ids:
        X_test = data[data['well_id'] == test_well_id][feature_names]
        Y_test = data[data['well_id'] == test_well_id][labels]
        Y_pred = model.pred(X_test)

        PLT().show2(Y_pred, Y_test, 'mlp', save_path=None)
        acc = accuracy_score(Y_pred, Y_test)
        res[test_well_id].append(np.round(acc, 2))
        tem_save[test_well_id] = "{:.1%}".format(acc)
    tem_save['other'] = other

    save.save(tem_save)
    return

verify(cur_well_ids, -1, 'initial')


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

        model = Model(model_name='mlp',
            type_name='classifier',
                         X_train=data[data['well_id'].isin(train_well_ids)][feature_names],
                         Y_train=data[data['well_id'].isin(train_well_ids)][labels],)
        Y_pred = model.pred(X_test)
        PLT().show2(Y_pred, Y_test, 'mlp', save_path=None)
        # 获取准确率
        acc = accuracy_score(Y_pred, Y_test)
        st.append((test_well_ids, acc))
        print(test_well_ids, acc)

    # st数组存储对实验结果,对他按照acc升序排序
    st.sort(key = lambda x : x[1])
    print(st)
    # 从井中剔除掉这些acc差的值，我们认为这个acc差的就是井数据噪声大的
    rem_well_ids = []
    for i in range(epoch_remove_cnt):
        print(st[i][0][0], ' had removed')
        rem_well_ids.append(st[i][0][0])
        cur_well_ids.remove(st[i][0][0])

    verify(cur_well_ids, epo, f'remove {rem_well_ids}, mlp')

print(res)

# # 画图
# plt = PLT(res, y_label='acc', x_label='epoch', xticks=list(range(1, 11)),
#           save_file=r'E:\项目\Intelligent_well_control\reports\images\overflow4.png')
# plt.show_acc()











