import joblib
import pandas as pd

model = joblib.load('../outputs/models/best_model.pkl')
model_columns = joblib.load('../outputs/models/model_columns.pkl')

def predict_churn(customer_dict):
    #convert input dict to single row dataframe
    input_df=pd.DataFrame([customer_dict])
    
    #rename col to match original training colum
    input_df.columns=model_columns
    
    churn_probability=model.predict_proba(input_df)[:,1][0]
    
    #categorise risk
    if churn_probability>=0.7:
        risk_category="High Risk"
    elif churn_probability>=0.4:
        risk_category="Medium Risk"
    else:
        risk_category="Low risk"
        
    return{
        "churn_probability":round(float(churn_probability),4),
        "risk_category":risk_category
    }
         
        
        
        