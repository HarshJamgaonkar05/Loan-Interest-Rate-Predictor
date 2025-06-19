from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd

df = pd.read_csv("Dataset.csv")

# Step 1: Separate features and target

target = "Interest Rate"
X = df.drop(columns=[target])
y = df[target]

# Step 2: Define feature types
numerical_cols = [
    "Loan Amount", "Income", "Credit Score", "Loan Term",
    "Property Value", "Debt-to-Income Ratio", "Loan-to-Value",
    "Income-to-Loan Ratio", "Monthly Payment", "Interest Burden", "Dependents"
]

categorical_cols = [
    "Credit Score Category", "Employment Type", "Loan Purpose", "Marital Status"
]

# Step 3: Create transformers
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Step 4: Build the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Step 5: Build preprocessing pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

# Step 6: Fit and transform
X_processed = pipeline.fit_transform(X)

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Step 8: Save the preprocessor and dataset splits
preprocessor_path = "preprocessor_pipeline.pkl"
dataset_split_path = "dataset_split.pkl"

joblib.dump(pipeline, preprocessor_path)
joblib.dump((X_train, X_test, y_train, y_test), dataset_split_path)

preprocessor_path, dataset_split_path
