import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Loading Model
model = tf.keras.models.load_model('breath.h5')

# Additional test, in case invalid value exists
for i in range(0,100):
    for j in range(0,100):
        input_data = np.array([[0, 0.01*i, 0.01*j]])
        prediction = model.predict(input_data)
        tick_length, inhale, exhale, repetition = prediction[0]
        if tick_length <= 0 or inhale <= 0 or exhale <= 0 or repetition <= 0:
            print("failure!")
            exit()
input_data = np.array([[0, 1, 0.2]])

# Prediction
prediction = model.predict(input_data)

# Result
tick_length, inhale,  exhale, repetition = prediction[0]
print(f"Tick length: {tick_length:.2f}")
print(f"Inhale: {inhale:.2f} ticks")
print(f"Exhale: {exhale:.2f} ticks")
print(f"Repetition: {repetition:.0f} times")
