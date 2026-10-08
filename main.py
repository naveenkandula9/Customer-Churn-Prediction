from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Customer Churn Prediction")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load the trained model
model = joblib.load("final_model.joblib")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    gender: str = Form(...),
    senior_citizen: int = Form(...),
    partner: str = Form(...),
    dependents: str = Form(...),
    tenure: int = Form(...),
    phone_service: str = Form(...),
    multiple_lines: str = Form(...),
    internet_service: str = Form(...),
    online_security: str = Form(...),
    online_backup: str = Form(...),
    device_protection: str = Form(...),
    tech_support: str = Form(...),
    streaming_tv: str = Form(...),
    streaming_movies: str = Form(...),
    contract: str = Form(...),
    paperless_billing: str = Form(...),
    payment_method: str = Form(...),
    monthly_charges: float = Form(...),
    total_charges: float = Form(...)
):
    # Encode the input data to match model features
    
    # Label encoding for binary columns
    gender_encoded = 1 if gender == "Male" else 0
    partner_encoded = 1 if partner == "Yes" else 0
    dependents_encoded = 1 if dependents == "Yes" else 0
    phone_service_encoded = 1 if phone_service == "Yes" else 0
    paperless_billing_encoded = 1 if paperless_billing == "Yes" else 0
    
    # One-hot encoding for MultipleLines
    multiple_lines_no_phone = (multiple_lines == "No phone service")
    multiple_lines_yes = (multiple_lines == "Yes")
    
    # One-hot encoding for InternetService
    internet_fiber = (internet_service == "Fiber optic")
    internet_no = (internet_service == "No")
    
    # One-hot encoding for OnlineSecurity
    online_security_no_internet = (online_security == "No internet service")
    online_security_yes = (online_security == "Yes")
    
    # One-hot encoding for OnlineBackup
    online_backup_no_internet = (online_backup == "No internet service")
    online_backup_yes = (online_backup == "Yes")
    
    # One-hot encoding for DeviceProtection
    device_protection_no_internet = (device_protection == "No internet service")
    device_protection_yes = (device_protection == "Yes")
    
    # One-hot encoding for TechSupport
    tech_support_no_internet = (tech_support == "No internet service")
    tech_support_yes = (tech_support == "Yes")
    
    # One-hot encoding for StreamingTV
    streaming_tv_no_internet = (streaming_tv == "No internet service")
    streaming_tv_yes = (streaming_tv == "Yes")
    
    # One-hot encoding for StreamingMovies
    streaming_movies_no_internet = (streaming_movies == "No internet service")
    streaming_movies_yes = (streaming_movies == "Yes")
    
    # One-hot encoding for Contract
    contract_one_year = (contract == "One year")
    contract_two_year = (contract == "Two year")
    
    # One-hot encoding for PaymentMethod
    payment_credit_card = (payment_method == "Credit card (automatic)")
    payment_electronic_check = (payment_method == "Electronic check")
    payment_mailed_check = (payment_method == "Mailed check")
    
    # Create DataFrame with all features in correct order
    input_data = pd.DataFrame({
        'gender': [gender_encoded],
        'SeniorCitizen': [senior_citizen],
        'Partner': [partner_encoded],
        'Dependents': [dependents_encoded],
        'tenure': [tenure],
        'PhoneService': [phone_service_encoded],
        'PaperlessBilling': [paperless_billing_encoded],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'MultipleLines_No phone service': [multiple_lines_no_phone],
        'MultipleLines_Yes': [multiple_lines_yes],
        'InternetService_Fiber optic': [internet_fiber],
        'InternetService_No': [internet_no],
        'OnlineSecurity_No internet service': [online_security_no_internet],
        'OnlineSecurity_Yes': [online_security_yes],
        'OnlineBackup_No internet service': [online_backup_no_internet],
        'OnlineBackup_Yes': [online_backup_yes],
        'DeviceProtection_No internet service': [device_protection_no_internet],
        'DeviceProtection_Yes': [device_protection_yes],
        'TechSupport_No internet service': [tech_support_no_internet],
        'TechSupport_Yes': [tech_support_yes],
        'StreamingTV_No internet service': [streaming_tv_no_internet],
        'StreamingTV_Yes': [streaming_tv_yes],
        'StreamingMovies_No internet service': [streaming_movies_no_internet],
        'StreamingMovies_Yes': [streaming_movies_yes],
        'Contract_One year': [contract_one_year],
        'Contract_Two year': [contract_two_year],
        'PaymentMethod_Credit card (automatic)': [payment_credit_card],
        'PaymentMethod_Electronic check': [payment_electronic_check],
        'PaymentMethod_Mailed check': [payment_mailed_check]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    churn_prob = probability[1] * 100
    stay_prob = probability[0] * 100
    
    result = {
        "prediction": "Will Churn" if prediction == 1 else "Will Stay",
        "churn_probability": round(churn_prob, 2),
        "stay_probability": round(stay_prob, 2),
        "is_churn": prediction == 1
    }
    
    # Prepare form data to repopulate form
    form_data = {
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges
    }
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "form_data": form_data
    })

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("Starting Customer Churn Prediction Server")
    print("="*50)
    print("\nOpen your browser and go to:")
    print("   http://localhost:8001")
    print("   or http://127.0.0.1:8001")
    print("\n" + "="*50 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8001)

