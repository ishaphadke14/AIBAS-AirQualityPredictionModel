import numpy as np
import pandas as pd
import tensorflow as tf
import pickle


tf.config.set_visible_devices([], 'GPU')


ACTIVATION_DATA_PATH = "/tmp/activationBase/activation_data.csv"
MODEL_PATH = "/tmp/knowledgeBase/currentAiSolution.keras"
SCALER_X_PATH = "/tmp/knowledgeBase/scaler_X.pkl"
SCALER_Y_PATH = "/tmp/knowledgeBase/scaler_y.pkl"

activation_data = pd.read_csv(ACTIVATION_DATA_PATH)

print("Activation data loaded")
#print("Columns:", activation_data.columns.tolist())
#print("Shape:", activation_data.shape)


if "aqi" in activation_data.columns:
    activation_data = activation_data.drop(columns=["aqi"])


with open(SCALER_X_PATH, "rb") as f:
    scaler_X = pickle.load(f)

with open(SCALER_Y_PATH, "rb") as f:
    scaler_y = pickle.load(f)

print("Scalers loaded successfully")

X_activation_scaled = scaler_X.transform(activation_data.values)
print("Activation input shape:", X_activation_scaled.shape)


model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("ANN model loaded")
print("Expected input shape:", model.input_shape)

y_pred_scaled = model.predict(X_activation_scaled)


y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()


results = activation_data.copy()
results["predicted_aqi"] = y_pred

print("Predicted AQI values:")
print(results[["predicted_aqi"]].head())

