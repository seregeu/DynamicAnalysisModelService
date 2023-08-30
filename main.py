from flask import Flask
from flask import request, jsonify

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, save_model, load_model
import json

app = Flask(__name__)

model_loaded = load_model('model.model')


@app.route("/dynamic_analyze", methods = ['POST'])
def hello_world():
    if request.method == 'POST':
        data = request.form
        #print(request.data)
        data_analysis(request.data)
        return jsonify(isError=False,
                       message="Success",
                       statusCode=200,
                       data=data), 200


def data_analysis(data):
    data_str = data.decode('utf-8')
    parsed_data = json.loads(data_str)
    df = pd.DataFrame(parsed_data)
    df = df.drop(['id', 'user_id', 'hit_x', 'hit_y', 'min_light', 'max_light'], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    X_reshaped = X_scaled[:, np.newaxis, :]
    predictions = model_loaded.predict(X_reshaped)
    predicted_labels = (predictions>0.5).astype(int)
    print(predicted_labels)
    print(df)
    pass


