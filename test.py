import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Loading Model
model = tf.keras.models.load_model('breath.h5')

# Modification of test data

input_data = np.array([[0, 1, 0.2]])

# Prediction
prediction = model.predict(input_data)

# Result
tick_length, inhale,  exhale, repetition = prediction[0]
print(f"Tick length: {tick_length:.2f}")
print(f"Inhale: {inhale:.2f} ticks")
print(f"Exhale: {exhale:.2f} ticks")
print(f"Repetition: {repetition:.0f} times")
