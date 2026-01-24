import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

tf.config.set_visible_devices([], 'GPU')  # Disable GPU

ACTIVATION_DATA_PATH = "/tmp/activationBase/activation_data.csv"
MODEL_PATH = "/tmp/knowledgeBase/currentAiSolution.h5"

# Read activation data
activation_data = pd.read_csv(ACTIVATION_DATA_PATH)

# Remove 'aqi' column if it exists (target variable shouldn't be in input)
if "aqi" in activation_data.columns:
    activation_data = activation_data.drop(columns=["aqi"])
   
scaler_X = joblib.load('/tmp/knowledgeBase/scaler_X.pkl')
X_activation = scaler_X.transform(activation_data.values)

print("Activation input shape:", X_activation.shape)

# Load model with safe_mode=False to handle version differences
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("ANN model loaded")
print("Expected input shape:", model.input_shape)

scaler_y = joblib.load('/tmp/knowledgeBase/scaler_y.pkl')
# Make predictions
y_pred = model.predict(X_activation).flatten()

pred_aqi = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Prepare results
results = activation_data.copy()
results["predicted_aqi"] = pred_aqi

print("Predicted AQI values:")
print(results[["predicted_aqi"]].head())

# If you want just the first prediction like your original code:
print(f"\nFirst Predicted AQI: {round(pred_aqi[0], 2)}")