from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Step 1: Load dataset_split.pkl
X_train, X_test, y_train, y_test = joblib.load("dataset_split.pkl")

# Step 2: Train the model
model = LGBMRegressor()
model.fit(X_train, y_train)

# Step 3: Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 4: Save model
joblib.dump(model, "model.pkl")
