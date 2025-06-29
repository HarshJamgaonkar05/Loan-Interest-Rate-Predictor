import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib


df = pd.read_csv("Dataset.csv")
X = df.drop(columns=["Interest Rate"])
y = df["Interest Rate"]


X = pd.get_dummies(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


joblib.dump(scaler, "Model/preprocessor_pipeline.pkl")
joblib.dump(X.columns.tolist(), "Model/expected_columns.pkl")


results = {}

#Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
preds_lr = lr.predict(X_test_scaled)
rmse_lr = mean_squared_error(y_test, preds_lr)
results["Linear Regression"] = (rmse_lr, lr)


# 2. Random Forest
rf = RandomForestRegressor(random_state=42)
rf_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None]
}

rf_cv = GridSearchCV(rf, rf_grid, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
rf_cv.fit(X_train_scaled, y_train)
best_rf = rf_cv.best_estimator_
preds_rf = best_rf.predict(X_test_scaled)
rmse_rf = mean_squared_error(y_test, preds_rf)
results["Random Forest"] = (rmse_rf, best_rf)


# 3. LightGBM
lgb = LGBMRegressor(random_state=42)
lgb_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "num_leaves": [31, 50]
}

lgb_cv = GridSearchCV(lgb, lgb_grid, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
lgb_cv.fit(X_train_scaled, y_train)
best_lgb = lgb_cv.best_estimator_
preds_lgb = best_lgb.predict(X_test_scaled)
rmse_lgb = mean_squared_error(y_test, preds_lgb)
results["LightGBM"] = (rmse_lgb, best_lgb)



print("Model Comparison (Lower RMSE is Better):")
for model, (rmse, _) in results.items():
    print(f"{model}: RMSE = {rmse:.4f}")

best_model_name = min(results, key=lambda x: results[x][0])
best_model = results[best_model_name][1]

# Save best model
joblib.dump(best_model, "Model/model.pkl")
print(f"\n Best model: {best_model_name} — saved to model.pkl")
