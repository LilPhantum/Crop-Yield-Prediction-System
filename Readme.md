# ClimateCropSim: Crop Yield Prediction Under Climate Variability

ClimateCropSim is a machine learning-based crop yield prediction system designed to estimate crop yield under projected climate conditions. The system uses rainfall, temperature, pesticide input, crop type, country selection, and land size to generate crop-specific yield predictions, visual charts, and AI-assisted recommendations.

## Overview

This project supports crop yield simulation for major crops used in the system: Maize, Rice, Sorghum, Cassava, Wheat, Yam, Potatoes, and Soybeans. The prediction engine is built with trained machine learning models and deployed through a web interface that allows users to run simulations without interacting directly with the model code.

The system focuses on selected African countries and was developed as a computational simulation system for predicting crop yield under climate variability.

## Key Features

- Crop yield prediction under projected 2026 climate conditions
- Crop-specific machine learning models
- Support for multiple crop selection
- Climate-based prediction using rainfall and temperature values
- Feature engineering for climate stress and interaction variables
- Total production calculation based on land size
- Interactive climate and yield visualizations using ApexCharts
- AI-based recommendation generated from completed prediction output
- FastAPI backend for API-based prediction
- Responsive frontend interface using HTML, CSS, and JavaScript

## Supported Crops

The system currently supports the following crops:

- Maize
- Rice
- Sorghum
- Cassava
- Wheat
- Yam
- Potatoes
- Soybeans

## Supported Countries

The deployed interface currently supports predictions for the following countries:

- Cameroon
- Ghana
- Kenya
- Mali
- Niger
- Senegal
- Burkina Faso
- South Africa
- Egypt
- Morocco

## Model Performance

The table below shows the available model evaluation results. The coefficient of determination, represented as R², measures how well the model explains crop yield variation during testing. A higher R² value indicates stronger predictive performance.

| Crop | Selected Model | Test R² | Performance Level |
|------|----------------|--------:|-------------------|
| Rice | Gradient Boosting | 0.9617 | Very Strong |
| Maize | Gradient Boosting | 0.9080 | Very Strong |
| Soybeans | Ensemble Model | 0.8524 | Strong |
| Wheat | Random Forest | 0.8157 | Strong |
| Potatoes | Ensemble Model | 0.7841 | Moderate to Strong |
| Cassava | Random Forest | 0.6728 | Moderate |
| Sorghum | Gradient Boosting | 0.6067 | Moderate |
| Yam | Crop Model | Pending | Awaiting final evaluation |

## 2026 Scenario Prediction Results

The system also supports scenario-based crop yield projection for 2026. The table below presents the baseline prediction, best case, worst case, and uncertainty range generated from the scenario analysis.

| Crop | 2026 Baseline Prediction | Best Case | Worst Case | Uncertainty Range |
|------|-------------------------:|----------:|-----------:|------------------:|
| Maize | 16,154 hg/ha | 17,297 hg/ha | 14,626 hg/ha | 2,671 hg/ha |
| Rice | 20,521 hg/ha | 22,842 hg/ha | 18,606 hg/ha | 4,236 hg/ha |
| Wheat | 17,127 hg/ha | 22,074 hg/ha | 16,948 hg/ha | 5,126 hg/ha |
| Cassava | 132,526 hg/ha | 140,274 hg/ha | 71,413 hg/ha | 68,860 hg/ha |
| Sorghum | 11,346 hg/ha | 12,034 hg/ha | 10,317 hg/ha | 1,717 hg/ha |
| Potatoes | 78,203 hg/ha | 86,643 hg/ha | 69,660 hg/ha | 16,984 hg/ha |
| Soybeans | 10,734 hg/ha | 13,568 hg/ha | 10,416 hg/ha | 3,152 hg/ha |

## Technology Stack

### Backend

- Python
- FastAPI
- Uvicorn
- Scikit-learn
- NumPy
- Pandas
- Joblib
- Google GenAI Library
- Python-dotenv

### Frontend

- HTML5
- CSS3
- JavaScript
- ApexCharts

### Data and Communication

- JSON for frontend and backend communication
- Serialized machine learning models stored as `.pkl` files
- Environment variable file for API key configuration

## Project Structure

```text
CropYieldPrediction/
├── backend/
│   ├── models/
│   │   └── trained crop model files
│   ├── server.py
│   ├── gemini.env
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   ├── script.js
│   ├── chart.js
│   └── images/
│
├── ml_scripts/
│   ├── training scripts
│   └── prediction scripts
│
└── README.md
```

## How the System Works

1. The user opens the web interface.
2. The user selects one or more crops.
3. The user selects a country.
4. The user enters land size in hectares.
5. The frontend validates the input.
6. The frontend sends the request to the FastAPI backend as JSON.
7. The backend retrieves climate values for the selected country.
8. The backend generates engineered prediction features.
9. The trained crop-specific model is loaded.
10. The model predicts yield per hectare.
11. The system calculates total production based on land size.
12. The AI recommendation module explains the completed output.
13. The frontend displays climate charts, yield charts, result summaries, and AI recommendation.

## Feature Engineering

The prediction system uses engineered features to improve model performance. These features help represent climate stress, crop response, and nonlinear relationships between variables.

Examples of engineered features include:

- Rainfall
- Temperature
- Pesticides
- Temperature-rainfall interaction
- Pesticide intensity
- Growing degree days
- Heat stress
- Cold stress
- Temperature deviation from 25°C
- Squared rainfall and temperature values
- Log rainfall and log pesticide values
- Rainfall category indicators
- Technology trend
- Years since 1990

## Machine Learning Methods

Different algorithms were selected for different crops based on their evaluation performance. The main machine learning methods used in the system include:

- Random Forest
- Gradient Boosting
- Ridge Regression
- Ensemble modelling

The system uses crop-specific modelling because different crops respond differently to rainfall, temperature, and agricultural input variables.

## AI-Based Recommendation

The system includes an AI-based recommendation component powered by Gemini API. This component receives the completed prediction output and generates a short advisory explanation for the user.

The AI recommendation does not perform the numerical crop yield prediction and does not modify the output of the trained machine learning models. Its purpose is to make the prediction result easier to understand.

## API Usage

### Prediction Endpoint

```http
POST /predict
Content-Type: application/json
```

### Sample Request

```json
{
  "crops": ["maize", "rice", "soybeans"],
  "country": "ghana",
  "land_size": 2.5
}
```

### Sample Response

```json
{
  "status": "success",
  "results": [
    {
      "crop": "Maize",
      "yield_per_ha": 16154,
      "total_production": 40385
    },
    {
      "crop": "Rice",
      "yield_per_ha": 20521,
      "total_production": 51302.5
    }
  ],
  "climate": {
    "rainfall": 1200,
    "temp": 26.8
  },
  "recommendation": "Rice shows the stronger projected yield under the selected climate and land size, while maize remains suitable under the same conditions."
}
```

## Installation and Setup

### Prerequisites

- Python 3.10 or higher
- pip
- Modern web browser
- Internet connection for Gemini API recommendation

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python server.py
```

The backend server runs on:

```text
http://localhost:3000
```

### Frontend Setup

Open the frontend in a browser, or serve it through the backend if the project is configured to expose the frontend files.

```text
frontend/index.html
```

## Environment Variables

The Gemini API key should be stored in an environment file instead of being written directly inside the source code.

Example `gemini.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

The backend should load the API key from the environment file before creating the Gemini client.

## Data Sources

The system was developed using agricultural and climate-related data. The major data source categories include:

- FAOSTAT agricultural data
- NASA POWER climate data
- Climate and crop yield records used for model training and evaluation

## Limitations

The system has the following limitations:

- It depends on historical data, so prediction quality depends on dataset accuracy and completeness.
- The deployed version uses selected climate variables and does not fully include soil fertility, fertilizer usage, irrigation, seed quality, pests, diseases, or farm management practices.
- Country-level climate values may not capture local variations within states, districts, or farming communities.
- The AI recommendation depends on an external API and requires internet access.
- Prediction results should be treated as decision support estimates, not guaranteed farm outcomes.

## Future Research Direction

Future versions of the system can be improved by expanding the prediction coverage to Nigeria, including all 36 states and the Federal Capital Territory. This would allow predictions to reflect local agricultural and climatic differences across Nigerian regions.

The system can also be improved by including more environmental and soil-related variables such as soil type, soil fertility, soil erosion, soil moisture, irrigation, fertilizer usage, humidity, solar radiation, pest occurrence, disease outbreak, and crop variety.

Another important future direction is the use of real-time data. Future versions can connect to weather APIs, satellite data, soil sensors, and remote sensing systems so that predictions are based on current and location-specific conditions rather than historical data alone.

A local AI recommendation system can also be developed for Nigerian agriculture. This would allow the system to provide more relevant recommendations based on local crops, local climate patterns, regional farming practices, and state-specific agricultural challenges.

## Disclaimer

AI-generated recommendations may contain errors and should not be treated as guaranteed agricultural advice. The prediction results are intended to support decision-making and should be used alongside agricultural knowledge, expert judgment, and local farming conditions.

## Author

Wakili Muazu Umar  
Matric Number: 22/03CMP039  
Department of Computer Science  
Faculty of Computing, Engineering and Technology  
Al-Hikmah University, Ilorin, Nigeria

## Project Title

Design and Implementation of a Computational Simulation System for Predicting Crop Yield Under Climate Variability
