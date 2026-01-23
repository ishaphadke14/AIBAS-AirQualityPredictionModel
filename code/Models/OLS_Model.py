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


train_df = pd.read_csv(r"..\..\data\processed\training_data.csv")
train_df.shape

y_train = train_df["aqi"]
X_train = train_df.drop(columns=["aqi"])

X_train = sm.add_constant(X_train)


ols_model = sm.OLS(y_train, X_train).fit()
print(ols_model.summary())


with open("currentOLSsolution.pkl", "wb") as file:
    pickle.dump(ols_model, file)

print("OLS model saved as 'currentOLSsolution.pkl'")


with open("currentOLSsolution.pkl", "rb") as file:
    loaded_model = pickle.load(file)

test_df = pd.read_csv(r"..\..\data\processed\test_data.csv")

X_test = test_df.drop(columns=["aqi"])
X_test = sm.add_constant(X_test)
y_test = test_df["aqi"]


y_pred = loaded_model.predict(X_test)
print(y_pred)


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
plt.title("Scatter Plot of Actual vs Predicted AQI (OLS)")
plt.grid(True)

plt.savefig("../../documentation/scatter_regression_plot_OLSModel.png", dpi=300)


residuals = y_test - y_pred
standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)


X = X_test.values
hat_matrix = X @ np.linalg.inv(X.T @ X) @ X.T
leverage = np.diag(hat_matrix)

cooks_d = (standardized_residuals ** 2) * leverage / (X.shape[1] * (1 - leverage) ** 2)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Residual vs Fitted
axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
axes[0, 0].axhline(y=0, color="r", linestyle="--")
axes[0, 0].set_title("Residual vs Fitted Values")
axes[0, 0].set_xlabel("Fitted Values")
axes[0, 0].set_ylabel("Residuals")

# 2. Scale-Location
sqrt_abs_resid = np.sqrt(np.abs(standardized_residuals))
axes[0, 1].scatter(y_pred, sqrt_abs_resid, alpha=0.5)
axes[0, 1].set_title("Scale-Location (Sqrt Standardized Residuals vs Fitted)")
axes[0, 1].set_xlabel("Fitted Values")
axes[0, 1].set_ylabel("√|Standardized Residuals|")

# 3. Q-Q plot
sm.qqplot(standardized_residuals, line="45", fit=True, ax=axes[1, 0])
axes[1, 0].set_title("Normal Q-Q Plot")
axes[1, 0].set_xlabel("Theoretical Quantiles")
axes[1, 0].set_ylabel("Standardized Residuals")

# 4. Residual vs Leverage
scatter = axes[1, 1].scatter(
    leverage,
    standardized_residuals,
    alpha=0.5,
    c=cooks_d,
    cmap="viridis"
)

axes[1, 1].set_title("Residual vs Leverage")
axes[1, 1].set_xlabel("Leverage")
axes[1, 1].set_ylabel("Standardized Residuals")


x = np.linspace(min(leverage), max(leverage), 50)
for c in [0.5, 1]:
    axes[1, 1].plot(
        x,
        np.sqrt((c * X.shape[1] * (1 - x) ** 2) / x),
        linestyle="--",
        color="red",
        label=f"Cook's D={c}"
    )
    axes[1, 1].plot(
        x,
        -np.sqrt((c * X.shape[1] * (1 - x) ** 2) / x),
        linestyle="--",
        color="red"
    )

axes[1, 1].legend()
fig.colorbar(scatter, ax=axes[1, 1], label="Cook's Distance")

plt.tight_layout()
plt.savefig("../../documentation/diagnostic_plots_OLSModel.png", dpi=300)
plt.show()
