import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Loading Model
model = tf.keras.models.load_model('breath.h5')

# Additional test, in case invalid value exists
test_inputs = np.array([[0, 0.01*i, 0.01*j] for i in range(100) for j in range(100)])
predictions = model.predict(test_inputs)
if np.any(predictions <= 0):
    problematic_indices = np.where(predictions <= 0)
    print("An invalid value is given in the index " + problematic_indices + "!")
    sys.exit(1)

# Prediction
prediction = model.predict(input_data)

# Result
tick_length, inhale,  exhale, repetition = prediction[0]
print(f"Tick length: {tick_length:.2f}")
print(f"Inhale: {inhale:.2f} ticks")
print(f"Exhale: {exhale:.2f} ticks")
print(f"Repetition: {repetition:.0f} times")
