import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import seaborn as sns
import pandas as pd


class PLT:
    def show_acc(self, x_label = None, y_label = None, data = None, xticks = None, save_file = None):
        plt.figure(figsize=(5, 3), dpi=200)
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        plt.xticks(xticks)
        plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])

        for name, values in data.items():
            plt.plot(list(range(1, len(values) + 1)), values, label=name)

        plt.legend()
        plt.show()
        plt.close()
        if save_file is not None:
            plt.savefig(save_file)
            print('image saved')


    def show_feature_plot(self, top = 40, model = None):
        feature_names = model.feature_name_
        feature_importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # 输出前 20 个特征
        top_features = feature_importance_df.head(top)

        plt.figure(figsize=(4, 6))
        sns.barplot(x='Importance', y='Feature', data=top_features, orient='h', palette='Set1')
        plt.title(f'Top {top} Feature Importances')
        plt.show()

    # 绘制每一个溢流判断与真实判断的线段图,线状散点图
    def show2(self, Y_pred, Y_true, title = 'test', save_path = None):

        plt.figure(figsize=(15, 1))  # 设置图形大小，确保每个像素点都能清晰显示
        # 根据预测结果设置每个像素点的颜色
        colors = ['g' if Y_true == Y_pred else 'r' for Y_true, Y_pred in zip(Y_true, Y_pred)]

        plt.scatter(range(len(Y_true)), np.ones_like(Y_true), c=colors, marker='|', s=200)

        # 找到第一个出现 1 的位置
        first_one_index = next((i for i, val in enumerate(Y_true) if val == 1), None)

        # 标记第一个出现 1 的位置
        if first_one_index is not None:
            plt.axvline(first_one_index, color='b', linestyle='--', linewidth=2)

        # 找到最后一个是1的索引
        last_one_index = next((len(Y_true) - i - 1 for i, val in enumerate(Y_true[::-1]) if val == 1), None)
        print(last_one_index, first_one_index)

        # 如果有值为1的元素，则计算在原始列表中的索引
        if last_one_index is not None:
            # last_one_index = len(Y_true) - last_one_index - 1
            plt.axvline(last_one_index, color='b', linestyle='--', linewidth=2)

        plt.yticks([])  # 隐藏y轴刻度
        plt.xlabel('index')
        plt.title(title)

        plt.show()

        if save_path != None:
            plt.savefig(save_path)
            print('save_plt_line!')





