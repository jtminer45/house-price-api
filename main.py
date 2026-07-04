from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define the API
app = FastAPI(title="House Price Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema
class HouseFeatures(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: float
    view: float
    condition: float
    grade: float
    sqft_above: float
    sqft_basement: float
    yr_built: float
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float
    house_age: float
    was_renovated: float
    price_per_sqft: float
    total_rooms: float

# Root endpoint
@app.get("/")
def home():
    return {"message": "House Price Predictor API", "status": "running"}

# Prediction endpoint
@app.post("/predict")
def predict(features: HouseFeatures):
    data = np.array([[
        features.bedrooms, features.bathrooms, features.sqft_living,
        features.sqft_lot, features.floors, features.waterfront,
        features.view, features.condition, features.grade,
        features.sqft_above, features.sqft_basement, features.yr_built,
        features.lat, features.long, features.sqft_living15,
        features.sqft_lot15, features.house_age, features.was_renovated,
        features.price_per_sqft, features.total_rooms
    ]])
    
    prediction = model.predict(data)
    
    return {
        "predicted_price": round(float(prediction[0]), 2),
        "currency": "USD"
    }