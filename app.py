import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Customer Churn Analysis", layout="centered")

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #020617, #0f172a 50%, #020617 100%);
}

h1, h2, h3 {
    text-align: center;
    color: #e5e7eb;
}

div, span, p {
    color: #e5e7eb;
}

div[data-testid="stMetric"] {
    background-color: #020617;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.6);
}

div[data-testid="stDataFrame"] {
    background-color: #020617;
    border-radius: 12px;
    padding: 10px;
}

pre {
    background-color: #020617;
    padding: 15px;
    border-radius: 10px;
    color: #e5e7eb;
}

button {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    border-radius: 10px;
    padding: 8px 20px;
    border: none;
}

button:hover {
    background: linear-gradient(90deg, #1d4ed8, #2563eb);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("📊 Customer Churn Prediction Dashboard")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn (1).csv")

df = load_data()

# -----------------------------
# Data Preprocessing
# -----------------------------
le = LabelEncoder()
df["Churn"] = le.fit_transform(df["Churn"])

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

X = df[["tenure", "MonthlyCharges", "TotalCharges"]]
y = df["Churn"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Model Training
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -----------------------------
# Model Evaluation
# -----------------------------
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
conf_matrix = confusion_matrix(y_test, y_pred)

# -----------------------------
# Predict for All Customers
# -----------------------------
X_scaled_full = scaler.transform(X)
df["Predicted_Churn"] = model.predict(X_scaled_full)

churn_count = (df["Predicted_Churn"] == 1).sum()
stay_count = (df["Predicted_Churn"] == 0).sum()

# -----------------------------
# Display Metrics
# -----------------------------
st.subheader("📌 Model Performance")
st.metric("Accuracy", f"{accuracy:.2f}")

st.subheader("📄 Classification Report")
st.text(class_report)

# -----------------------------
# Visual Confusion Matrix
# -----------------------------
st.subheader("📊 Confusion Matrix")

fig, ax = plt.subplots()
cm_display = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix,
    display_labels=["No Churn", "Churn"]
)
cm_display.plot(ax=ax, cmap="Blues")
st.pyplot(fig)

# -----------------------------
# Churn Summary
# -----------------------------
st.subheader("🚨 Churn Prediction Summary")

c1, c2, c3 = st.columns(3)
c1.metric("Total Customers", len(df))
c2.metric("Customers Likely to Leave", churn_count)
c3.metric("Customers Likely to Stay", stay_count)

# -----------------------------
# Sample Predictions
# -----------------------------
st.subheader("📋 Sample Customer Predictions")
st.dataframe(
    df[["tenure", "MonthlyCharges", "TotalCharges", "Predicted_Churn"]]
    .replace({"Predicted_Churn": {0: "No", 1: "Yes"}})
    .head(10)
)

# -----------------------------
# User Prediction Section
# -----------------------------
st.markdown("---")
st.subheader("🔮 Predict Customer Churn (New Customer)")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
total = st.slider("Total Charges", 0.0, 10000.0, 2000.0)

if st.button("🚀 Predict Churn"):
    input_df = pd.DataFrame([[tenure, monthly, total]], columns=X.columns)
    input_data = scaler.transform(input_df)
    probability = model.predict_proba(input_data)[0][1]

    if probability >= 0.5:
        st.error(f"⚠️ Likely to Leave (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Likely to Stay (Probability: {probability:.2f})")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Logistic Regression | Streamlit | Customer Churn Analysis")
