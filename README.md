# Customer Churn Prediction System

## Project Overview
The Customer Churn Prediction System is a machine learning-based application that predicts whether a telecom customer is likely to leave a service.

The project uses customer information, service subscriptions, contract details, tenure, and billing information to predict churn.

## Objectives
- Analyze customer data to identify churn patterns.
- Clean and preprocess the dataset.
- Perform feature engineering.
- Train and compare multiple machine learning models.
- Evaluate models using Accuracy, Precision, Recall, ROC-AUC, and PR-AUC.
- Deploy the trained model using a FastAPI REST API.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- FastAPI
- Uvicorn
- Joblib
- VS Code

## Machine Learning Models
Three classification algorithms are trained and compared:
1. Logistic Regression
2. Random Forest
3. Decision Tree

The current model-selection rule selects the model with the highest ROC-AUC score.

## Dataset
The project uses the IBM Telco Customer Churn dataset.

- Total records: 7,043
- Original columns: 21
- Columns after feature engineering: 25

## Feature Engineering
Four additional features are created:
- TenureGroup
- TotalServices
- IsLongTermCustomer
- AverageMonthlySpend

## Model Evaluation
The models are evaluated using:
- Accuracy
- Precision
- Recall
- ROC-AUC
- PR-AUC

The selected Logistic Regression model achieved:
- Accuracy: 73.39%
- Precision: 49.92%
- Recall: 79.14%
- ROC-AUC: 0.8408
- PR-AUC: 0.6309

## Project Structure

```text
Customer_Churn_Prediction/
│
├── data/
│   └── telco_churn.csv
│
├── models/
│   └── best_churn_model.joblib
│
├── venv/
│
├── train.py
├── api.py
├── requirements.txt
└── README.md
```

## Installation and Execution

### 1. Clone the repository
```bash
git clone https://github.com/divyanka22/Customer-Churn-Prediction
cd Customer_Churn_Prediction
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Start the API
```bash
uvicorn api:app --reload
```

### 6. Open API documentation

http://127.0.0.1:8000/docs

## API Endpoint

**POST /predict**

Accepts customer details and returns:
- Churn prediction (Yes/No)
- Churn probability

**GET /**

Checks whether the API is running.

## Conclusion
The project demonstrates a complete customer churn prediction workflow, from data preprocessing and feature engineering to model training, evaluation, and REST API deployment.