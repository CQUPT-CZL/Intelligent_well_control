from flask import Flask, jsonify, request
import joblib
# import lightgbm as lgb
from io import StringIO
import pandas as pd

app = Flask(__name__)

model = joblib.load('lgb_model.pkl')

def pred(X):
    Y_pred = model.pred(X)
    print("pred：", Y_pred)
    return str(Y_pred)

@app.route('/test', methods=['GET'])
def test():
    return 'hello'

@app.route('/get_pred', methods=['POST'])
def get_pred():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # 检查文件名是否为空
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # 检查文件类型（可选）
    if file.mimetype != 'text/csv':
        return jsonify({'error': 'Invalid file type'})

    # 读取上传的 CSV 文件
    csv_data = file.read().decode('utf-8')

    df = pd.read_csv(StringIO(csv_data))

    return jsonify({'status': 'success', 'pred': pred(df)})


if __name__ == '__main__':
    app.run(debug=True)
