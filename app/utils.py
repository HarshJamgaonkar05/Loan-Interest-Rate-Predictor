import pandas as pd
import joblib
import os

# Load model and preprocessor
model = joblib.load(os.path.join("Model", "model.pkl"))
pipeline = joblib.load(os.path.join("Model", "preprocessor_pipeline.pkl"))

# Define fields
NUM_FIELDS = ["Loan Amount", "Income", "Credit Score", "Loan Term", "Property Value",
              "Debt-to-Income Ratio", "Dependents"]
CAT_FIELDS = ["Credit Score Category", "Employment Type", "Loan Purpose", "Marital Status"]

def predict_interest_rate(form_data):
    input_data = {}

    for field in NUM_FIELDS:
        input_data[field] = float(form_data.get(field, 0))

    for field in CAT_FIELDS:
        input_data[field] = form_data.get(field, "")

    # Derived features
    input_data["Loan-to-Value"] = input_data["Loan Amount"] / input_data["Property Value"]
    input_data["Income-to-Loan Ratio"] = input_data["Income"] / input_data["Loan Amount"]
    r = 0.04 / 12
    n = input_data["Loan Term"] * 12
    input_data["Monthly Payment"] = input_data["Loan Amount"] * r * (1 + r)**n / ((1 + r)**n - 1)
    input_data["Interest Burden"] = input_data["Monthly Payment"] / (input_data["Income"] / 12)

    df = pd.DataFrame([input_data])
    transformed = pipeline.transform(df)
    return model.predict(transformed)[0]

def explain_prediction(form_data):
    features = {}

    # Extract inputs
    for field in NUM_FIELDS:
        features[field] = float(form_data.get(field, 0))
    for field in CAT_FIELDS:
        features[field] = form_data.get(field, '')

    # Derived metrics
    ltv = features["Loan Amount"] / features["Property Value"]
    income_to_loan = features["Income"] / features["Loan Amount"]
    monthly_income = features["Income"] / 12
    r = 0.04 / 12
    n = features["Loan Term"] * 12
    emi = features["Loan Amount"] * r * (1 + r)**n / ((1 + r)**n - 1)
    interest_burden = emi / monthly_income

    reasons = []

    # Credit Score
    credit = features["Credit Score"]
    if credit < 600:
        reasons.append(
            f" Credit Score : Your credit score is {credit}, which is considered poor. "
            "Scores under 600 suggest high default risk, which likely increased your interest rate."
        )
    elif credit >= 750:
        reasons.append(
            f" Credit Score** : Your credit score is {credit}, which is considered excellent. "
            "Lenders trust high scores, so this likely lowered your rate."
        )
    else:
        reasons.append(
            f" Credit Score** : Your credit score is {credit}, which is within an average range. "
            "This probably had a neutral or slight effect on your interest rate."
        )

    # LTV
    if ltv > 0.8:
        reasons.append(
            f" Loan-to-Value (LTV) Ratio : Your LTV ratio is {ltv:.2f}, meaning you're borrowing a large share of the property value. "
            "Since LTV > 0.80, lenders may see this as higher risk, contributing to a higher interest rate."
        )
    else:
        reasons.append(
            f" Loan-to-Value (LTV) Ratio : Your LTV ratio is {ltv:.2f}. Staying under 0.80 is generally viewed as low risk, which likely helped keep your interest rate lower."
        )

    # Debt-to-Income Ratio
    dti = features["Debt-to-Income Ratio"]
    if dti > 0.4:
        reasons.append(
            f" Debt-to-Income (DTI) Ratio : Your DTI is {dti:.2f}, which is considered high. "
            "High DTI ratios signal potential repayment difficulty and likely increased your interest rate."
        )
    else:
        reasons.append(
            f" Debt-to-Income (DTI) Ratio : Your DTI is {dti:.2f}, within acceptable limits. This likely had a neutral or positive impact."
        )

    # Employment Type
    emp = features["Employment Type"]
    if emp == "Unemployed":
        reasons.append(
            " Employment Type : You're currently listed as Unemployed. "
            "Lenders prefer steady income sources. This likely increased your interest rate."
        )
    elif emp == "Salaried":
        reasons.append(
            " Employment Type : You're listed as Salaried. "
            "Stable income from salaried employment is preferred and may have lowered your interest rate."
        )
    else:
        reasons.append(
            f" Employment Type : You're listed as {emp}. "
            "Lenders may view this as moderate-risk, with a small effect on your rate."
        )

    # Income-to-Loan Ratio
    if income_to_loan < 0.3:
        reasons.append(
            f" Income-to-Loan Ratio : Your ratio is {income_to_loan:.2f}, which is considered low. "
            "This means you're borrowing a large amount compared to your income, which may have increased your rate."
        )
    else:
        reasons.append(
            f" Income-to-Loan Ratio : Your ratio is {income_to_loan:.2f}, which is healthy. "
            "This likely helped secure a better interest rate."
        )

    return reasons
