from fastapi import FastAPI
from schema import Customer
from model_utils import predict_churn

app=FastAPI(title="Customer Churn Prediction API")

@app.get("/")
def home():
    return {"message":"Churn Prediction API is running"}

@app.post("/predict")
def predict(customer:Customer):
    customer_dict=customer.dict()
    result=predict_churn(customer_dict)
    return result
