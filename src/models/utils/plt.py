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

    # 绘制每一个溢流判断与真实判断的线段图,真实的，预测中，交叉的
    def show2(self, Y_pred, Y_true, title, save_path = None):

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10))

        # plt.figure(figsize=(40, 5))  # 设置图形大小，确保每个像素点都能清晰显示

        # 根据预测结果设置每个像素点的颜色

        colors = ['g' if i == 0 else 'r' for i in Y_true]
        axes[0].scatter(range(len(Y_true)), np.ones_like(Y_true), c=colors, marker='|', s=200)
        axes[0].set_yticks([])  # 隐藏y轴刻度
        axes[0].set_xlabel('index')
        axes[0].set_title('Y_true')

        colors = colors = ['g' if i == 0 else 'r' for i in Y_pred]
        axes[1].scatter(range(len(Y_true)), np.ones_like(Y_true), c=colors, marker='|', s=200)
        axes[1].set_yticks([])  # 隐藏y轴刻度
        axes[1].set_xlabel('index')
        axes[1].set_title('Y_pred')

        colors = ['g' if Y_true == Y_pred else 'r' for Y_true, Y_pred in zip(Y_true, Y_pred)]
        axes[2].scatter(range(len(Y_true)), np.ones_like(Y_true), c=colors, marker='|', s=200)
        axes[2].set_yticks([])  # 隐藏y轴刻度
        axes[2].set_xlabel('index')
        axes[2].set_title('true&pred')

        # 调整布局
        plt.tight_layout()

        # plt.show()

        if save_path is not None:

            plt.savefig(save_path)
            print('save_plt_line_successfully')





