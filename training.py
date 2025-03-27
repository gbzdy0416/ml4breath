import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

#Loading data

data = pd.read_csv('things.csv',
                   names=["function", "intensity_function", "intensity_time",
                          "tick_length", "inhale", "hold", "exhale", "repetition"])

#Setting training set

input_set = data[["function", "intensity_function", "intensity_time"]].values
output_set = data[["tick_length", "inhale", "hold", "exhale", "repetition"]].values
input_train, input_test, output_train, output_test = train_test_split(input_set, output_set,
                                                                      test_size=0.2, random_state=42)
#Setting and Training Model

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5)
])
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
model.fit(input_train, output_train, epochs=50, batch_size=16, validation_data=(input_test, output_test))

#Converting into Concrete Function

run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, input_train.shape[1]], model.inputs[0].dtype))

#Saving Model
model.save('breath.h5')