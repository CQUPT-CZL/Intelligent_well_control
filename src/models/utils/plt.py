import matplotlib.pyplot as plt
from datetime import datetime


class PLT:
    def __init__(self, data, y_label, x_label, xticks = None, yticks = None, save_file = None):
        self.data = data
        self.y_label = y_label
        self.x_label = x_label
        self.xticks = xticks
        self.yticks = yticks
        self.save_file = save_file


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


