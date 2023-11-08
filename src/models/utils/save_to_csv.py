'''
将一个字典类型的数据保存到一个给定的csv中
'''
import pandas as pd
import csv

class SaveToCsv():
    def __init__(self, csv_file = 'results.csv'):
        self.csv_file = csv_file

    def save(self, result):
        fieldnames = list(result.keys())
        try:
            with open(self.csv_file, 'r') as file:
                # 文件已存在，不需要写表头
                pass
        except FileNotFoundError:
            with open(self.csv_file, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()

        # 打开CSV文件，追加新的结果
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow(result)

        print('successfully save')
        return
