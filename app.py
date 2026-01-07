import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("📊 Telco Customer Churn Prediction")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("Telco-Customer-Churn.csv")

df = load_data()

# ===============================
# PREPROCESSING (SAME AS TRAINING)
# ===============================
df["Churn"] = df["Churn"].str.strip().str.lower().map({"yes": 1, "no": 0})
df = df.dropna(subset=["Churn"])

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df = df.drop("customerID", axis=1)

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])

le = LabelEncoder()
label_encoders = {}
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(solver="liblinear", max_iter=200)
model.fit(X_train, y_train)

# ===============================
# INPUT FORM (FEATURE TRACKING)
# ===============================
st.subheader("🔍 Enter Customer Details")

tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

gender = st.selectbox("Gender", ["Male", "Female"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

# ===============================
# CREATE INPUT DATAFRAME
# ===============================
input_dict = {
    "gender": gender,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "InternetService": internet_service,
    "Contract": contract,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

input_df = pd.DataFrame([input_dict])

# Encode categorical inputs
for col in input_df.columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

# Match training column order
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Scale
input_scaled = scaler.transform(input_df)

# ===============================
# PREDICTION
# ===============================
if st.button("Predict Churn"):
    prob = model.predict_proba(input_scaled)[0][1]

    st.write(f"### 🔢 Churn Probability: **{prob:.2f}**")

    if prob >= 0.5:
        st.error("⚠️ Customer is **Likely to Churn**")
    else:
        st.success("✅ Customer is **Likely to Stay**")