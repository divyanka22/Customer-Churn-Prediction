
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

# Create FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a telecom customer is likely to churn.",
    version="1.0"
)

# Load the saved model
model = joblib.load("models/best_churn_model.joblib")


# Customer input structure
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# Home route
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running!"
    }


# Prediction route
@app.post("/predict")
def predict_churn(customer: CustomerData):

    customer_data = customer.model_dump()

    # Feature engineering
    tenure = customer_data["tenure"]

    if tenure <= 12:
        tenure_group = "New"
    elif tenure <= 24:
        tenure_group = "Short-term"
    elif tenure <= 48:
        tenure_group = "Medium-term"
    else:
        tenure_group = "Long-term"

    service_columns = [
        "PhoneService",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    total_services = (
        int(customer_data["PhoneService"] == "Yes")
        + int(customer_data["InternetService"] != "No")
        + sum(
            customer_data[col] == "Yes"
            for col in service_columns[2:]
        )
    )

    is_long_term = int(tenure >= 24)

    if tenure > 0:
        average_monthly_spend = (
            customer_data["TotalCharges"] / tenure
        )
    else:
        average_monthly_spend = customer_data["MonthlyCharges"]

    # Add engineered features
    customer_data["TenureGroup"] = tenure_group
    customer_data["TotalServices"] = total_services
    customer_data["IsLongTermCustomer"] = is_long_term
    customer_data["AverageMonthlySpend"] = average_monthly_spend

    # Convert input into DataFrame
    input_data = pd.DataFrame([customer_data])

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4)
    }
