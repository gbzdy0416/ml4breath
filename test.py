import numpy as np
import tensorflow as tf
import sys
import os

def start_test(input_path='BreathingModel.h5'):
    # Loading Model
    print("Loading Model...")
    if not os.path.exists(input_path):
        print(f"File does not exist：{input_path}")
        sys.exit(1)
    try:
        model = tf.keras.models.load_model(input_path)
    except Exception as e:
        print(f"Model loading failed: {e}")
        sys.exit(1)

    print("Test of correctness of the model...")
    # Pre-defined denormalizer
    output_min = np.array([746, 3, 5, 4])  # tick_length, inhale, exhale, repetition
    output_max = np.array([1250, 9, 10, 9])
    # Additional test, in case invalid value exists
    test_inputs = np.array([[0, 0.01*i, 0.01*j] for i in range(100) for j in range(100)])
    predictions = model.predict(test_inputs)
    for predict in predictions:
        if predict[0].any() < 0:
            print("There are several invalid predictions generated by the model!")
            sys.exit(1)

    print("No invalid values are given. Regular test starting...")
    input_data = np.array([[0,1.2,0]])
    # Prediction
    prediction = model.predict(input_data)

    # Result
    tick_length, inhale,  exhale, repetition = prediction[0] * (output_max - output_min) + output_min
    print(f"Tick length: {tick_length:.2f}")
    print(f"Inhale: {inhale:.2f} ticks")
    print(f"Exhale: {exhale:.2f} ticks")
    print(f"Repetition: {repetition:.0f} times")


if __name__ == "__main__":
    start_test()