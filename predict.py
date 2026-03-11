from zenml.client import Client
import pandas as pd
from fastapi import FastAPI
import uvicorn
import numpy as np

app = FastAPI(title="Smart Real Estate Predictor")

def get_latest_trained_model():
    """Fetches the model artifact from the latest successful pipeline run."""
    client = Client()
    # Grabs the latest successful run
    last_run = client.get_pipeline("smart_real_estate_pipeline").runs[0]
    
    # Access the step, then the output, then LOAD the data
    # .load() is the correct method for ArtifactVersionResponse objects
    model = last_run.steps["model_trained"].output.load()
    return model
@app.get("/")
def home():
    return {
        "message": "Smart Real Estate Predictor API is LIVE!",
        "version": "1.0",
        "instructions": "Send a POST request to /predict with house features to get a price."
    }
@app.post("/predict")
def predict_price(input_data: dict):
    # 1. Load the model
    model = get_latest_trained_model()
    
    # 2. Get the exact list of 38 features the model was trained on
    expected_features = model.feature_names_in_
    
    # 3. Create a template DataFrame with one row of zeros
    # This ensures the 'shape' (1, 38) is exactly what the Random Forest expects
    df = pd.DataFrame(0, index=[0], columns=expected_features)
    
    # 4. Fill the template with the data you actually sent in the curl
    for key, value in input_data.items():
        if key in expected_features:
            df[key] = value
        else:
            print(f"Note: '{key}' is not used by this model.")
            
    # 5. Run the prediction
    prediction = model.predict(df)
    
    return {
        "status": "success",
        "predicted_sale_price": round(float(prediction[0]), 2),
        "currency": "USD"
    }

if __name__ == "__main__":
    # Start the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)