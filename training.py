import keras
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from losses import custom_loss
from sklearn.preprocessing import RobustScaler
import os
import sys
import joblib

def load_and_prepare_data(csv_path='data.csv'):
    #Loading data
    if not os.path.exists(csv_path):
        print(f"Data file not found: {csv_path}")
        sys.exit(1)

    #Splitting data into input and output attributes
    df = pd.read_csv(csv_path, names=["intensity_function", "intensity_time",
                                      "inhale", "exhale", "repetition", "reduction", "satisfaction", "age_group", "gender"])
    features = df[['intensity_function', 'intensity_time', "age_group", "gender"]]
    labels = df[['inhale', 'exhale', 'repetition', "reduction", "satisfaction"]]

    #Using a scaler to normalize the input data
    scaler = RobustScaler()
    input_set = scaler.fit_transform(features)
    output_scaler = RobustScaler()
    output_scaled = output_scaler.fit_transform(labels)
    joblib.dump(scaler, 'scaler.save')
    joblib.dump(output_scaler, 'y_scaler.save')
    print("Scalers are saved")
    return train_test_split(input_set, output_scaled, test_size=0.2, random_state=42)

def build_model(input_train):
    #To build the model...
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(16, activation='tanh'),
        tf.keras.layers.Dense(5, activation='tanh')
    ])
    model.compile(
        optimizer='adam',
        loss=custom_loss,
        metrics=['mae']
    )
    return model

def train_model():
    #Getting the data for training
    print("Start loading the data...")
    input_train, input_test, output_train, output_test = load_and_prepare_data()

    #Building and training the model
    print("Data prepared. Start training model...")
    model = build_model(input_train)
    model.fit(input_train, output_train, batch_size=16, epochs=100)
    model.save('BreathingModel_2.h5')
    print("Model trained and saved as BreathingModel_2.h5, scalers are also saved")

if __name__ == '__main__':
    train_model()
