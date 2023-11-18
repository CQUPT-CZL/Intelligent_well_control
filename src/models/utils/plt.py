import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


class PLT:
    def __init__(self, a):
        self.a = a
        # self.data = data
        # self.y_label = y_label
        # self.x_label = x_label
        # self.xticks = xticks
        # self.yticks = yticks
        # self.save_file = save_file


    def show_acc(self):
        plt.figure(figsize=(5, 3), dpi=200)
        plt.ylabel(self.y_label)
        plt.xlabel(self.x_label)

        plt.xticks(self.xticks)
        plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])

        for name, values in self.data.items():
            plt.plot(list(range(1, len(values) + 1)), values, label=name)

        plt.legend()
        plt.show()
        plt.close()
        if self.save_file is not None:
            plt.savefig(self.save_file)
            print('image saved')

    # 绘制每一个溢流判断与真实判断的线段图
    def show2(self, Y_pred, Y_true, title, save_path = None):

        plt.figure(figsize=(15, 1))  # 设置图形大小，确保每个像素点都能清晰显示
        # 根据预测结果设置每个像素点的颜色
        colors = ['g' if Y_true == Y_pred else 'r' for Y_true, Y_pred in zip(Y_true, Y_pred)]

        plt.scatter(range(len(Y_true)), np.ones_like(Y_true), c=colors, marker='|', s=200)

        plt.yticks([])  # 隐藏y轴刻度
        plt.xlabel('index')
        plt.title(title)

        # plt.show()

        if save_path != None:

            plt.savefig(save_path)
            print('save_plt_line_successfully')





