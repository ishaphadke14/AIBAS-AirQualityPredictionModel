import os
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import numpy as np


# ------------------------------
# Paths
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
DOCS_DIR = os.path.join(BASE_DIR, 'documentation')
MODELS_DIR = os.path.join(BASE_DIR, 'code', 'Models')

train_path = os.path.join(DATA_DIR, 'training_data.csv')
test_path  = os.path.join(DATA_DIR, 'test_data.csv')

# ------------------------------
# Load data
# ------------------------------
train_data = pd.read_csv(train_path)
test_data  = pd.read_csv(test_path)

X_train = train_data.drop('aqi', axis=1)
y_train = train_data['aqi']

X_test = test_data.drop('aqi', axis=1)
y_test = test_data['aqi']

# ------------------------------
# Standardize numeric features
# ------------------------------
numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

# ------------------------------
# Add constant for intercept
# ------------------------------
X_train_sm = sm.add_constant(X_train)
X_test_sm  = sm.add_constant(X_test)

# ------------------------------
# Fit OLS model
# ------------------------------
ols_model = sm.OLS(y_train, X_train_sm).fit()

# ------------------------------
# Predictions
# ------------------------------
y_pred = ols_model.predict(X_test_sm)

# ------------------------------
# Metrics
# ------------------------------
mse  = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
os.makedirs(DOCS_DIR, exist_ok=True)
with open(os.path.join(DOCS_DIR, 'ols_performance_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

print("OLS Model metrics:", metrics)

# ------------------------------
# Save OLS model as pickle
# ------------------------------
os.makedirs(MODELS_DIR, exist_ok=True)
with open(os.path.join(MODELS_DIR, 'currentOLSsolution.pkl'), 'wb') as f:
    pickle.dump(ols_model, f)
print("OLS model saved at:", os.path.join(MODELS_DIR, 'currentOLSsolution.pkl'))

# ------------------------------
# Diagnostic Plots
# ------------------------------
residuals = y_test - y_pred
standardized_residuals = (residuals - residuals.mean()) / residuals.std()

# Leverage & Cook's distance
X = sm.add_constant(np.column_stack([np.ones_like(y_pred), y_pred]))
hat_matrix = X @ np.linalg.inv(X.T @ X) @ X.T
leverage = np.diag(hat_matrix)
cooks_d = (standardized_residuals**2) * leverage / (X.shape[1] * (1 - leverage)**2)

# Residuals vs Fitted
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Fitted Values'); plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.grid(True)

# Scale-Location
plt.subplot(2, 2, 2)
plt.scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
plt.xlabel('Fitted Values'); plt.ylabel('√|Standardized Residuals|')
plt.title('Scale-Location')
plt.grid(True)

# Q-Q plot
plt.subplot(2, 2, 3)
sm.qqplot(standardized_residuals, line='45', fit=True, alpha=0.5, ax=plt.gca())
plt.title('Normal Q-Q Plot')

# Residuals vs Leverage
plt.subplot(2, 2, 4)
scatter = plt.scatter(leverage, standardized_residuals, alpha=0.5, c=cooks_d, cmap='viridis')
plt.xlabel('Leverage'); plt.ylabel('Standardized Residuals')
plt.title('Residuals vs Leverage')
cbar = plt.colorbar(scatter)
cbar.set_label("Cook's Distance")

plt.tight_layout()
plt.savefig(os.path.join(DOCS_DIR, 'ols_diagnostic_plots.png'), dpi=300)
plt.close()
print("Diagnostic plots saved at:", os.path.join(DOCS_DIR, 'ols_diagnostic_plots.png'))

# ------------------------------
# Scatter plot: Predicted vs Actual
# ------------------------------
plt.figure(figsize=(8,6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ci=None)
plt.xlabel('Actual AQI'); plt.ylabel('Predicted AQI'); plt.title('Predicted vs Actual AQI')
plt.grid(True)
plt.savefig(os.path.join(DOCS_DIR, 'ols_scatter_regression.png'), dpi=300)
plt.close()
print("Scatter regression plot saved at:", os.path.join(DOCS_DIR, 'ols_scatter_regression.png'))
