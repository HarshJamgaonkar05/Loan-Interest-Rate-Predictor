from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd

df = pd.read_csv("Dataset.csv")



target = "Interest Rate"
X = df.drop(columns=[target])
y = df[target]


numerical_cols = ["Loan Amount", "Income", "Credit Score", "Loan Term","Property Value", "Debt-to-Income Ratio", "Loan-to-Value","Income-to-Loan Ratio", 
                  "Monthly Payment", "Interest Burden", "Dependents"]

categorical_cols = ["Credit Score Category", "Employment Type", "Loan Purpose", "Marital Status"]


numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")


preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numerical_cols),("cat", categorical_transformer, categorical_cols),])


pipeline = Pipeline(steps=[("preprocessor", preprocessor)])


X_processed = pipeline.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)


preprocessor_path = "preprocessor_pipeline.pkl"
dataset_split_path = "dataset_split.pkl"

joblib.dump(pipeline, preprocessor_path)
joblib.dump((X_train, X_test, y_train, y_test), dataset_split_path)

preprocessor_path, dataset_split_path
