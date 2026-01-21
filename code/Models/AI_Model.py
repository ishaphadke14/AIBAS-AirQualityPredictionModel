import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json

# Load data
train_data = pd.read_csv('D:/AIBAS-AirQualityIndexPrediction/AIBAS-AirQualityPredictionModel/code/DatasetScraping&Cleaning/data/processed/training_data.csv')
test_data = pd.read_csv('D:/AIBAS-AirQualityIndexPrediction/AIBAS-AirQualityPredictionModel/code/DatasetScraping&Cleaning/data/processed/test_data.csv')

X_train_raw = train_data.drop('aqi', axis=1).values
y_train_raw = train_data['aqi'].values

X_test_raw = test_data.drop('aqi', axis=1).values
y_test_raw = test_data['aqi'].values

# Scale features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train_raw)
X_test = scaler_X.transform(X_test_raw)

y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()

# Build ANN model
def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    return model

model = build_model(X_train.shape[1])

# Train model with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=150,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# Save model
os.makedirs('models', exist_ok=True)
model.save('models/currentAiSolution.keras')

# Inverse transform predictions
y_pred_scaled = model.predict(X_test).flatten()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = y_test_raw  # Original AQI

# Calculate metrics
mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
with open('models/ai_performance_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# -----------------------------
# Training curves
# -----------------------------
def plot_training_curves(history):
    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], label='Train Loss')
    plt.plot(epochs, history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs'); plt.ylabel('MSE'); plt.title('Training Curves - Loss')
    plt.legend(); plt.grid(True)

    # MAE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['mae'], label='Train MAE')
    plt.plot(epochs, history.history['val_mae'], label='Val MAE')
    plt.xlabel('Epochs'); plt.ylabel('MAE'); plt.title('Training Curves - MAE')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    
    plt.savefig('D:/AIBAS-AirQualityIndexPrediction/AIBAS-AirQualityPredictionModel/documentation/ai_training_curves.png', dpi=300)
    plt.close()

plot_training_curves(history)

# -----------------------------
# Regression Diagnostic Plots
# -----------------------------
def plot_diagnostic_plots(y_test, y_pred):
    residuals = y_test - y_pred
    standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

    # Simple leverage example using hat matrix
    X = sm.add_constant(np.column_stack([np.ones_like(y_pred), y_pred]))
    hat_matrix = X @ np.linalg.inv(X.T @ X) @ X.T
    leverage = np.diag(hat_matrix)

    cooks_d = (standardized_residuals ** 2) * leverage / (X.shape[1] * (1 - leverage) ** 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Residual vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].set_xlabel('Fitted Values'); axes[0, 0].set_ylabel('Residuals')

    # Scale-Location
    axes[0, 1].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
    axes[0, 1].set_title('Scale-Location (Sqrt Standardized Residuals)')
    axes[0, 1].set_xlabel('Fitted Values'); axes[0, 1].set_ylabel('√|Standardized Residuals|')

    # Q-Q plot
    sm.qqplot(standardized_residuals, line='45', fit=True, alpha=0.5, ax=axes[1, 0])
    axes[1, 0].set_title('Normal Q-Q Plot')

    # Residual vs Leverage with Cook's distance
    scatter = axes[1, 1].scatter(leverage, standardized_residuals, alpha=0.5, c=cooks_d, cmap='viridis')
    axes[1, 1].set_title('Residuals vs Leverage')
    axes[1, 1].set_xlabel('Leverage'); axes[1, 1].set_ylabel('Standardized Residuals')
    cbar = fig.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label("Cook's Distance")

    plt.tight_layout()
    plt.savefig('D:/AIBAS-AirQualityIndexPrediction/AIBAS-AirQualityPredictionModel/documentation/ai_diagnostic_plots.png', dpi=300)
    plt.close()

plot_diagnostic_plots(y_test_actual, y_pred)

# -----------------------------
# Predicted vs Actual Scatter with Regression
# -----------------------------
def plot_scatter_with_regression(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ci=None)
    plt.xlabel('Actual AQI'); plt.ylabel('Predicted AQI'); plt.title('Predicted vs Actual AQI')
    plt.grid(True)
    
    plt.savefig('D:/AIBAS-AirQualityIndexPrediction/AIBAS-AirQualityPredictionModel/documentation/ai_scatter_regression.png', dpi=300)
    plt.close()

plot_scatter_with_regression(y_test_actual, y_pred)

print("✅ Plots saved: training curves, diagnostic plots, scatter regression plot")
