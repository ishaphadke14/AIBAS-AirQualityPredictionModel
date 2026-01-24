import os
import pandas as pd
import statsmodels.api as sm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
DOCS_DIR = os.path.join(BASE_DIR, "documentation")
MODELS_DIR = os.path.join(BASE_DIR, "code", "Models")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# Load data

train_df = pd.read_csv(os.path.join(DATA_DIR, "training_data.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))

X_train = train_df.drop("aqi", axis=1)
y_train = train_df["aqi"]

X_test = test_df.drop("aqi", axis=1)
y_test = test_df["aqi"]

# Add constant
X_train = sm.add_constant(X_train)
X_test  = sm.add_constant(X_test)


# Train OLS model

ols_model = sm.OLS(y_train, X_train).fit()
#print(ols_model.summary())


# Predictions & metrics

y_pred = ols_model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"R²   : {r2:.3f}")


# Save model

with open(os.path.join(MODELS_DIR, "currentOLSsolution.pkl"), "wb") as f:
    pickle.dump(ols_model, f)


# Linear Regression Plot

plt.figure(figsize=(8, 6))
sns.regplot(
    x=y_test,
    y=y_pred,
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red"},
    ci=None
)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("OLS Regression: Actual vs Predicted AQI")
plt.grid(True)

plt.savefig(os.path.join(DOCS_DIR, "ols_actual_vs_predicted.png"), dpi=300)
plt.close()


# Diagnostic Plots
y_train_pred = ols_model.predict(X_train)

train_residuals = y_train - y_train_pred
std = train_residuals.std() if train_residuals.std() != 0 else 1
standardized_residuals = (train_residuals - train_residuals.mean()) / std

leverage = ols_model.get_influence().hat_matrix_diag

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Residuals vs Fitted
axes[0, 0].scatter(y_train_pred, train_residuals, alpha=0.5)
axes[0, 0].axhline(0, color="red", linestyle="--")
axes[0, 0].set_title("Residuals vs Fitted")
axes[0, 0].set_xlabel("Fitted Values")
axes[0, 0].set_ylabel("Residuals")

# 2. Scale-Location
axes[0, 1].scatter(
    y_train_pred,
    np.sqrt(np.abs(standardized_residuals)),
    alpha=0.5
)
axes[0, 1].set_title("Scale-Location")
axes[0, 1].set_xlabel("Fitted Values")
axes[0, 1].set_ylabel("√|Standardized Residuals|")

# 3. Q-Q Plot
sm.qqplot(standardized_residuals, line="45", fit=True, ax=axes[1, 0])
axes[1, 0].set_title("Normal Q-Q")

# 4. Residuals vs Leverage
axes[1, 1].scatter(leverage, standardized_residuals, alpha=0.5)
axes[1, 1].set_title("Residuals vs Leverage")
axes[1, 1].set_xlabel("Leverage")
axes[1, 1].set_ylabel("Standardized Residuals")

plt.tight_layout()
plt.savefig(os.path.join(DOCS_DIR, "ols_diagnostic_plots.png"), dpi=300)
plt.close()


print("\nOLS model executed successfully.")
print("Plots saved in documentation/")
