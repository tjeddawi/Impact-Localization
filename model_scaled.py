import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

# Data Preprocessing
data = pd.read_csv("sensor_data.csv")

X = data.drop(columns=['X', 'Y', 'Force'])
y = data[['X', 'Y', 'Force']]

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

best_params = {
    'n_estimators': 500,
    'max_depth': 4,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'gamma': 0.01,
    'alpha': 1,
    'lambda': 0
}
xgb_model = xgb.XGBRegressor(**best_params)

xgb_model.fit(X_train_scaled, y_train)

y_test_pred_scaled = xgb_model.predict(X_test_scaled)

# Calculate RMSE for each coordinate and force value
rmse_x_test = mean_squared_error(y_test['X'], y_test_pred_scaled[:, 0], squared=False)
rmse_y_test = mean_squared_error(y_test['Y'], y_test_pred_scaled[:, 1], squared=False)
rmse_force_test = mean_squared_error(y_test['Force'], y_test_pred_scaled[:, 2], squared=False)

print(f"Testing Set X RMSE: {rmse_x_test:.3f}")
print(f"Testing Set Y RMSE: {rmse_y_test:.3f}")
print(f"Testing Set Force RMSE: {rmse_force_test:.3f}")

plt.figure(figsize=(15, 5))

# X Coordinate
plt.subplot(1, 3, 1)
plt.scatter(y_test['X'], y_test_pred_scaled[:, 0], alpha=0.5)
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, linestyle='--', color='black')  # y = x line
plt.xlabel('Actual X')
plt.ylabel('Predicted X')
plt.title('Predicted vs. Actual X Coordinate')
plt.text(0.05, 0.9, f'RMSE: {rmse_x_test:.2f}', transform=plt.gca().transAxes)

# Y Coordinate
plt.subplot(1, 3, 2)
plt.scatter(y_test['Y'], y_test_pred_scaled[:, 1], alpha=0.5)
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, linestyle='--', color='black')  # y = x line
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.title('Predicted vs. Actual Y Coordinate')
plt.text(0.05, 0.9, f'RMSE: {rmse_y_test:.2f}', transform=plt.gca().transAxes)

# Force Value
plt.subplot(1, 3, 3)
plt.scatter(y_test['Force'], y_test_pred_scaled[:, 2], alpha=0.5)
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, linestyle='--', color='black')  # y = x line
plt.xlabel('Actual Force')
plt.ylabel('Predicted Force')
plt.title('Predicted vs. Actual Force Value')
plt.text(0.05, 0.9, f'RMSE: {rmse_force_test:.2f}', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()
