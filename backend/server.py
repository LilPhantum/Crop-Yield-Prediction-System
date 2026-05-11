from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from google import genai
import joblib
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore', message='X does not have valid feature names')

load_dotenv("gemini.env")

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

# Gemini client
gemini_client = genai.Client()

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
        pred = (
            w['rf'] * model_data['rf_model'].predict(X)[0] +
            w['gb'] * model_data['gb_model'].predict(X)[0] +
            w['ridge'] * model_data['ridge_model'].predict(
                model_data['ridge_scaler'].transform(X)
            )[0]
        )

    elif model_data['type'] == 'Ridge':
        X_scaled = model_data['model']['scaler'].transform(X)
        pred = model_data['model']['model'].predict(X_scaled)[0]

    else:
        pred = model_data['model'].predict(X)[0]

    return pred


def limit_text(text, max_chars=700):
    text = text.strip()

    if len(text) > max_chars:
        return text[:697].rstrip() + "..."

    return text


def fallback_recommendation(country, land_size, climate, results):
    best_crop = max(results, key=lambda item: item["yield_per_ha"])
    country_name = country.replace("_", " ").title()

    text = (
        f"{best_crop['crop']} is the strongest option in this simulation, "
        f"with the highest predicted yield of {best_crop['yield_per_ha']:,.0f} hg/ha. "
        f"For {land_size} hectares in {country_name}, prioritize this crop while monitoring rainfall "
        f"and temperature. This is a 2026 simulation result, so it should guide planning, not guarantee output."
    )

    return limit_text(text)


def generate_ai_recommendation(country, land_size, climate, results):
    prompt = f"""
You are an agricultural simulation assistant.

Use only the simulation data provided below.
Write one practical crop recommendation for a farmer.
Maximum length: 700 characters.
Use one short paragraph only.
Do not use bullet points, markdown, headings, or long explanations.
Recommend only from the selected crops.
Mention the strongest crop based on predicted yield.
Mention climate suitability using rainfall or temperature if useful.
Do not make guarantees.
Do not invent facts.
Do not recommend crops outside the selected crops.

Simulation data:
Country: {country.replace("_", " ").title()}
Land size: {land_size} hectares
Projected year: 2026
Rainfall: {climate["rainfall"]} mm
Temperature: {climate["temp"]} °C
Results: {results}
"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        if not response.text:
            return fallback_recommendation(country, land_size, climate, results)

        return limit_text(response.text)

    except Exception as e:
        print(f"Gemini recommendation error: {e}")
        return fallback_recommendation(country, land_size, climate, results)


@app.get("/")
async def root():
    return {
        "message": "Crop Yield Prediction API",
        "frontend": "/ui/index.html"
    }


@app.post("/predict")
async def predict_yield(req: Request):
    if req.country not in COUNTRIES:
        raise HTTPException(
            400,
            f"Invalid country. Available: {list(COUNTRIES.keys())}"
        )

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

    recommendation = generate_ai_recommendation(
        country=req.country,
        land_size=req.land_size,
        climate=climate,
        results=results
    )

    return {
        'status': 'success',
        'results': results,
        'climate': {
            'rainfall': climate['rainfall'],
            'temp': climate['temp']
        },
        'recommendation': recommendation
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)