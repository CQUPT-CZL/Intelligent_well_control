'''
将总的数据处理文件中处理完成后的数据（一半总数据）
再进行细分，只把那些区块井大于等于3的区块数据存储来
'''
import pandas as pd

class Processing2():

    def __init__(self, data = None):
        self.data = data

    def find(self, cnt = 3):
        block_ids = self.data['block_id'].unique().tolist()

        st = {}
        for block_id in block_ids:
            well_ids = self.data[self.data['block_id'] == block_id]['well_id'].unique().tolist()
            if len(well_ids) >= cnt:
                st[block_id] = well_ids

        return st

    def save(self, path = r'E:\data\压井\新数据\间接数据'):
        block_ids = self.find().keys()
        print(block_ids)
        self.data[self.data['block_id'].isin(block_ids)].to_csv(path + "\大区块数据2.csv", index = False)

if __name__ == '__main__':
    path = r'E:\data\压井\新数据\间接数据\一半总数据2.csv'
    Processing2(pd.read_csv(path)).save()

