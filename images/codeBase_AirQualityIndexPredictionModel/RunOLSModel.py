import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle


MODEL_PATH = "/tmp/knowledgeBase/currentOLSsolution.pkl"

with open(MODEL_PATH, "rb") as f:
    ols_model = pickle.load(f)

print("OLS model loaded successfully")


ACTIVATION_DATA_PATH = "/tmp/activationBase/activation_data.csv"
activation_data = pd.read_csv(ACTIVATION_DATA_PATH)

print("Activation data loaded")
print(activation_data.head())


if "aqi" in activation_data.columns:
    activation_data = activation_data.drop(columns=["aqi"])

activation_data = sm.add_constant(activation_data, has_constant="add")

print("Activation data shape after preprocessing:", activation_data.shape)


predictions = ols_model.predict(activation_data)

output = activation_data.copy()
output["predicted_aqi"] = predictions

print("Predictions completed")
print(output[["predicted_aqi"]].head())

print(f"\nFirst Predicted AQI: {round(predictions[0], 2)}")


