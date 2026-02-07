"""Final Working Backend with Frontend Serving"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore', message='X does not have valid feature names')

app = FastAPI()

# Serve frontend
app.mount("/ui", StaticFiles(directory="../frontend", html=True), name="frontend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Country climate data
COUNTRIES = {
    'cameroon': {'rainfall': 1600, 'temp': 24.5},
    'ghana': {'rainfall': 1200, 'temp': 26.8},
    'kenya': {'rainfall': 1100, 'temp': 19.5},
    'mali': {'rainfall': 750, 'temp': 28.5},
    'niger': {'rainfall': 500, 'temp': 29.0},
    'senegal': {'rainfall': 650, 'temp': 28.0},
    'burkina_faso': {'rainfall': 900, 'temp': 28.5},
    'south_africa': {'rainfall': 600, 'temp': 18.5},
    'egypt': {'rainfall': 50, 'temp': 22.5},
    'morocco': {'rainfall': 350, 'temp': 17.5}
}

CROPS = {
    'maize': 'Maize',
    'rice': 'Rice_paddy',
    'sorghum': 'Sorghum',
    'cassava': 'Cassava',
    'wheat': 'Wheat',
    'yam': 'Yams',
    'potatoes': 'Potatoes',
    'soybeans': 'Soybeans'
}

class Request(BaseModel):
    crops: List[str]
    country: str
    land_size: float


def make_features(rain, temp, pest=6000):
    f = {
        'rainfall': rain,
        'temperature': temp,
        'pesticides': pest,
        'temp_rain_interaction': temp * rain / 1000,
        'pesticide_intensity': pest / (rain + 1),
        'growing_degree_days': max(temp - 10, 0) * 365,
        'heat_stress': max(temp - 30, 0),
        'cold_stress': max(10 - temp, 0),
        'temp_deviation_from_25': abs(temp - 25),
        'temperature_sq': temp ** 2,
        'rainfall_sq': rain ** 2,
        'pesticides_sq': pest ** 2,
        'log_pesticides': np.log1p(pest),
        'log_rainfall': np.log1p(rain),
        'years_since_1990': 36,
        'tech_trend': 6.0,
        'rain_cat_dry': 1 if rain < 500 else 0,
        'rain_cat_moderate': 1 if 500 <= rain < 1000 else 0,
        'rain_cat_wet': 1 if 1000 <= rain < 1500 else 0,
        'rain_cat_very_wet': 1 if rain >= 1500 else 0
    }
    return f


def predict_crop(crop_name, rain, temp):
    model_path = f"models/model_{crop_name}.pkl"
    model_data = joblib.load(model_path)
    
    features = make_features(rain, temp)
    X = np.array([[features.get(f, 0) for f in model_data['features']]])
    
    if model_data['type'] == 'Ensemble':
        w = model_data['weights']
        pred = (w['rf'] * model_data['rf_model'].predict(X)[0] +
                w['gb'] * model_data['gb_model'].predict(X)[0] +
                w['ridge'] * model_data['ridge_model'].predict(
                    model_data['ridge_scaler'].transform(X))[0])
    elif model_data['type'] == 'Ridge':
        X_scaled = model_data['model']['scaler'].transform(X)
        pred = model_data['model']['model'].predict(X_scaled)[0]
    else:
        pred = model_data['model'].predict(X)[0]
    
    return pred


@app.get("/")
async def root():
    return {"message": "Crop Yield Prediction API", "frontend": "/ui/index.html"}


@app.post("/predict")
async def predict_yield(req: Request):
    if req.country not in COUNTRIES:
        raise HTTPException(400, f"Invalid country. Available: {list(COUNTRIES.keys())}")
    
    climate = COUNTRIES[req.country]
    results = []
    
    for crop in req.crops:
        if crop not in CROPS:
            continue
        try:
            y = predict_crop(CROPS[crop], climate['rainfall'], climate['temp'])
            results.append({
                'crop': crop.title(),
                'yield_per_ha': round(y, 2),
                'total_production': round(y * req.land_size, 2)
            })
        except Exception as e:
            print(f"Error predicting {crop}: {e}")
            continue
    
    if not results:
        raise HTTPException(400, "No valid predictions")
    
    avg = np.mean([r['yield_per_ha'] for r in results])
    
    return {
        'status': 'success',
        'results': results,
        'climate': {
            'rainfall': climate['rainfall'],
            'temp': climate['temp']
        },
        'recommendation': f"Based on climate data from {req.country.title()}, expected average yield is {avg:,.0f} hg/ha. Climate conditions are suitable for selected crops. Ensure proper agricultural inputs for optimal results."
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)