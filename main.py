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
model_loaded_batch = load_model('model_batch_size1.model')


@app.route("/dynamic_analyze", methods = ['POST'])
def hello_world():
    if request.method == 'POST':
        data = request.form
        #print(request.data)
        print(">>>Regular analysis")
        data_analysis(request.data)
        print(">>>Batch analysis")
        data_analysis_batch(request.data)
        #print(">>>Legacy analysis")
        #data_analysis_legacy(request.data)
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
    sequence = []
    i=0
    for i in range(len(X_reshaped)):
        sequence.append(X_reshaped[i])
    sequence_np = np.array(sequence)
    predictions = model_loaded.predict(sequence_np)
    predicted_labels = (predictions > 0.5).astype(int)
    print(predictions)
    print(df)
    pass

def data_analysis_batch(data):
    model_loaded_batch.reset_states()
    data_str = data.decode('utf-8')
    parsed_data = json.loads(data_str)
    df = pd.DataFrame(parsed_data)
    df = df.drop(['id', 'user_id', 'hit_x', 'hit_y', 'min_light', 'max_light'], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    X_reshaped = X_scaled[:, np.newaxis, :]

    num_features = X_scaled.shape[1]
    print("num_featires = ", num_features)
    batch_size = 1
    timesteps = 1
    new_data_reshaped = X_scaled.reshape(-1, timesteps, num_features)


    sequence = []
    i=0
    for i in range(len(new_data_reshaped)):
        sequence.append(new_data_reshaped[i])
    sequence_np = np.array(sequence)
    predictions = model_loaded_batch.predict(sequence_np, batch_size=batch_size)
    predicted_labels = (predictions > 0.5).astype(int)
    print(predictions)
    print(df)
    pass

def data_analysis_legacy(data):
    data_str = data.decode('utf-8')
    parsed_data = json.loads(data_str)
    df = pd.DataFrame(parsed_data)
    df = df.drop(['id', 'user_id', 'hit_x', 'hit_y', 'min_light', 'max_light'], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    X_reshaped = X_scaled[:, np.newaxis, :]
    predictions = model_loaded.predict(X_reshaped)
    predicted_labels = (predictions>0.5).astype(int)
    print(predictions)
    print(df)
    pass