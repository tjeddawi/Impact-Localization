import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from scipy.stats import uniform
import matplotlib.pyplot as plt

# Data Preprocessing

train_data = pd.read_csv("training_set.csv")
val_data = pd.read_csv("validation_set.csv")
test_data = pd.read_csv("testing_set.csv")

X_train = train_data.drop(columns=['Number', 'X Coordinate', 'Y Coordinate', 'Force Value'])
y_train = train_data[['X Coordinate', 'Y Coordinate', 'Force Value']]

X_val = val_data.drop(columns=['Number', 'X Coordinate', 'Y Coordinate', 'Force Value'])
y_val = val_data[['X Coordinate', 'Y Coordinate', 'Force Value']]

X_test = test_data.drop(columns=['Number', 'X Coordinate', 'Y Coordinate', 'Force Value'])
y_test = test_data[['X Coordinate', 'Y Coordinate', 'Force Value']]

xgb_model = xgb.XGBRegressor()

# Param Grid
"""param_dist = {
    'n_estimators': range(50, 500, 50),
    'max_depth': range(3, 11),
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
    'subsample': uniform(0.5, 0.5),
    'gamma': [0, 1, 2, 3],
    'alpha': uniform(0, 10),
    'lambda': uniform(0, 10)
}"""
best_params = {
    'alpha': 0.4530400977204452,
    'gamma': 2,
    'lambda': 0.1530454029038475,
    'learning_rate': 0.1,
    'max_depth': 4,
    'n_estimators': 50,
    'subsample': 0.7696887242000312
}
random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=100, cv=3, random_state=42)

random_search.fit(X_train, y_train)
best_model = xgb.XGBRegressor(**best_params)


y_test_pred = best_model.predict(X_test)

rmse_x_test = mean_squared_error(y_test['X Coordinate'], y_test_pred[:, 0], squared=False)
rmse_y_test = mean_squared_error(y_test['Y Coordinate'], y_test_pred[:, 1], squared=False)
rmse_force_test = mean_squared_error(y_test['Force Value'], y_test_pred[:, 2], squared=False)

print(f"Testing Set X RMSE: {rmse_x_test:.3f}")
print(f"Testing Set Y RMSE: {rmse_y_test:.3f}")
print(f"Testing Set Force RMSE: {rmse_force_test:.3f}")

plt.figure(figsize=(15, 5))

# X Coordinate
plt.subplot(1, 3, 1)
plt.scatter(y_test['X Coordinate'], y_test_pred[:, 0], alpha=0.5)
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, linestyle='--', color='black')  # y = x line
plt.xlabel('Actual X')
plt.ylabel('Predicted X')
plt.title('Predicted vs. Actual X Coordinate')
plt.text(0.05, 0.9, f'RMSE: {rmse_x_test:.2f}', transform=plt.gca().transAxes)

# Y Coordinate
plt.subplot(1, 3, 2)
plt.scatter(y_test['Y Coordinate'], y_test_pred[:, 1], alpha=0.5)
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, linestyle='--', color='black')  # y = x line
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.title('Predicted vs. Actual Y Coordinate')
plt.text(0.05, 0.9, f'RMSE: {rmse_y_test:.2f}', transform=plt.gca().transAxes)

# Force Value
plt.subplot(1, 3, 3)
plt.scatter(y_test['Force Value'], y_test_pred[:, 2], alpha=0.5)
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, linestyle='--', color='black')  # y = x line
plt.xlabel('Actual Force')
plt.ylabel('Predicted Force')
plt.title('Predicted vs. Actual Force Value')
plt.text(0.05, 0.9, f'RMSE: {rmse_force_test:.2f}', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()
