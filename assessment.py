import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# 10 models...
model_files = [
    'BreathingModel_0.keras',
    'BreathingModel_1.keras',
    'BreathingModel_2.keras',
    'BreathingModel_3.keras',
    'BreathingModel_4.keras',
    'BreathingModel_5.keras',
    'BreathingModel_6.keras',
    'BreathingModel_7.keras',
    'BreathingModel_8.keras',
    'BreathingModel_9.keras',
]

# loading Scaler
scaler = joblib.load('scaler.save')
output_scaler = joblib.load('y_scaler.save')

# loading test input
df = pd.read_csv("testinput.csv", header=None)
X_input = scaler.transform(df.iloc[:, 0:4].values)

# making prediction
all_preds = []
for model_path in model_files:
    model = tf.keras.models.load_model(model_path, compile=False)
    preds = model.predict(X_input)
    all_preds.append(preds)

# denormalization
all_preds = np.array(all_preds)  # shape: [10 models, 20 samples, 5 outputs]
all_preds_scaled = all_preds.reshape(-1, 5)
all_preds_real = output_scaler.inverse_transform(all_preds_scaled)
all_preds = all_preds_real.reshape(10, -1, 5)  # [10, 20, 5]

# mean and std
mean_preds = np.mean(all_preds, axis=0)    # shape: [20 samples, 5 outputs]
std_matrix = np.std(all_preds, axis=0)      # shape: [20 samples, 5 outputs]

# cv calculation
cv_matrix = std_matrix / (np.abs(mean_preds) + 1e-8) * 100   # 加1e-8防止除零错误

output_labels = ['Inhalation', 'Exhalation', 'Repetition']
num_outputs = len(output_labels)

# figure drawing
cvs_per_output = []
for i in range(num_outputs):
    cvs = cv_matrix[:, i]  # shape: [20 samples]
    cvs_per_output.append(cvs)

plt.figure(figsize=(12, 6))
plt.boxplot(cvs_per_output, patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color='darkblue'),
            widths=0.6)

plt.xticks(np.arange(1, num_outputs + 1), output_labels, rotation=15, fontsize=14)
plt.ylabel("Coefficient of Variation (CV) [%]", fontsize=14)
plt.title("Prediction Stability Across Models (CV Analysis)", fontsize=18)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()

# mean cv calculation
mean_cv_per_dim = np.mean(cv_matrix, axis=0)  # shape: [5 outputs]
print("Mean CV per output dimension (inhale, exhale, repeat, satisfaction, effectiveness):")
print(mean_cv_per_dim)

# mean max difference calculation
max_diff_matrix = np.max(all_preds, axis=0) - np.min(all_preds, axis=0)  # shape: [20 samples, 5 outputs]
mean_max_diff_per_dim = np.mean(max_diff_matrix, axis=0)
print("Mean max difference per output dimension (inhale, exhale, repeat, satisfaction, effectiveness):")
print(mean_max_diff_per_dim)
