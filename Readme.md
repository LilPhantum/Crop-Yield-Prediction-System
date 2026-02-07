# Crop Yield Prediction System Under Climate Variability

A machine learning-based system for predicting crop yields across West Africa using ensemble modeling and climate data.

## 🌾 Overview

This system predicts crop yields for 8 major crops (Maize, Rice, Sorghum, Cassava, Wheat, Yam, Potatoes, Soybeans) using climate variables (temperature, rainfall) and agricultural inputs (pesticides as proxy for farm management). The models are trained on multi-country African agricultural data and deployed via a web interface for interactive predictions.

## 📊 Key Features

- **High Accuracy:** Average R² = 0.81 across all crops (range: 0.61-0.96)
- **Ensemble Models:** Combines Random Forest, Gradient Boosting, and Ridge Regression
- **Cross-Country Training:** 12 African countries, 2,776 samples (1990-2013)
- **Advanced Feature Engineering:** 20+ features including climate interactions and stress indicators
- **Web Interface:** Interactive prediction system with ApexCharts visualizations
- **Climate Scenarios:** Analyze yield under different climate conditions (baseline, warming, drought, etc.)

## 🎯 Performance

| Crop | Model | Test R² | Status |
|------|-------|---------|--------|
| Rice | Gradient Boosting | 0.96 | ✓✓ Outstanding |
| Maize | Gradient Boosting | 0.91 | ✓✓ Excellent |
| Sweet Potatoes | Gradient Boosting | 0.88 | ✓✓ Excellent |
| Soybeans | Ensemble | 0.85 | ✓✓ Excellent |
| Wheat | Random Forest | 0.82 | ✓✓ Excellent |
| Potatoes | Ensemble | 0.78 | ✓✓ Excellent |
| Cassava | Random Forest | 0.67 | ✓ Good |
| Sorghum | Gradient Boosting | 0.61 | ✓ Good |

## 🛠️ Technology Stack

**Backend:**
- Python 3.8+
- FastAPI (REST API)
- scikit-learn (ML models)
- NumPy, Pandas (data processing)
- joblib (model serialization)

**Frontend:**
- HTML5, CSS3, JavaScript
- ApexCharts (data visualization)
- Responsive design

**Data Sources:**
- FAO Agricultural Database
- World Bank Climate Data
- NASA POWER Climate Archive

## 📁 Project Structure

```
CropYieldPrediction/
├── backend/
│   ├── models/                  # Trained ML models (.pkl files)
│   ├── data/
│   │   └── yield_df.csv        # Training dataset
│   ├── server.py               # FastAPI backend
│   └── requirements.txt
│
├── frontend/
│   ├── index.html              # Main interface
│   ├── script.js               # Application logic
│   ├── charts.js               # ApexCharts visualizations
│   └── styles.css              # Styling
│
└── ml_scripts/
    ├── train.py                # Model training script
    └── predict.py              # Prediction utilities
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/crop-yield-prediction.git
cd crop-yield-prediction
```

2. **Install dependencies**
```bash
cd backend
pip install -r requirements.txt
```

3. **Start the backend server**
```bash
python server.py
```
Server runs on: `http://localhost:3000`

4. **Access the frontend**
- Option 1: Navigate to `http://localhost:3000/ui/index.html`
- Option 2: Open `frontend/index.html` directly in browser

## 📝 Usage

### Web Interface

1. Select crops from dropdown (up to 8 crops)
2. Choose country from West African region
3. Enter land size in hectares
4. Click "Run Simulation"
5. View results:
   - Climate data visualization (rainfall & temperature)
   - Crop yield predictions (bar chart)
   - Detailed yield breakdown
   - AI-generated recommendations

### API Endpoint

```python
POST /predict
Content-Type: application/json

{
  "crops": ["maize", "rice", "sorghum"],
  "country": "ghana",
  "land_size": 2.5
}

Response:
{
  "status": "success",
  "results": [
    {
      "crop": "Maize",
      "yield_per_ha": 18317.11,
      "total_production": 45792.78
    }
  ],
  "climate": {
    "rainfall": 1200,
    "temp": 26.8
  },
  "recommendation": "..."
}
```

## 🔬 Methodology

### Feature Engineering
The system uses 20+ engineered features:
- **Core:** rainfall, temperature, pesticides
- **Interactions:** temp_rain_interaction, pesticide_intensity
- **Climate Stress:** growing_degree_days, heat_stress, cold_stress
- **Polynomials:** temperature_sq, rainfall_sq, pesticides_sq
- **Transformations:** log_pesticides, log_rainfall
- **Temporal:** years_since_1990, tech_trend
- **Categorical:** rain_cat_dry, rain_cat_moderate, rain_cat_wet, rain_cat_very_wet

### Model Training
- **Training Period:** 1990-2010
- **Testing Period:** 2011-2013
- **Validation:** Temporal splitting (prevents data leakage)
- **Ensemble Strategy:** Weighted averaging by validation R²

### Countries in Training Data
Cameroon, Ghana, Kenya, Mali, Niger, Senegal, Burkina Faso, South Africa, Egypt, Morocco, Algeria, Tunisia

## 📊 Supported Countries

Current predictions available for:
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

## 🌍 Climate Scenarios

The system supports 11 climate scenarios:
1. Baseline (2026 Trends)
2. Optimistic (Good Practices)
3. Moderate Warming (+1°C)
4. High Warming (+2°C)
5. Severe Warming (+3°C)
6. Moderate Drought (-15% rain)
7. Severe Drought (-30% rain)
8. Wet Conditions (+20% rain)
9. Hot & Dry (+2°C, -20% rain)
10. Extreme Stress (+3°C, -30% rain)
11. Best Management (+inputs, +rain)

## 📈 Key Findings

1. **Agricultural inputs > Climate:** Pesticide intensity is the #1 or #2 most important feature for most crops
2. **Crop-specific vulnerabilities:** Cassava shows -46% yield under hot+dry conditions
3. **Non-linear responses:** Rice benefits from moderate drought (reduces flooding), Wheat prefers hot+dry
4. **Management potential:** Good agricultural practices can improve yields 4-16%, offsetting moderate climate stress

## 🔒 Limitations

- Trained on historical data (1990-2013); may not capture unprecedented climate conditions
- Pesticides used as proxy for overall farm management (not perfect)
- Does not account for: pests, diseases, extreme events, soil degradation, socio-economic factors
- Regional models may not capture highly localized conditions

## 🛣️ Future Work

- [ ] Add real-time weather data integration
- [ ] Incorporate soil quality data
- [ ] Extend to more African countries
- [ ] Add mobile application
- [ ] Include economic analysis (cost-benefit)
- [ ] Integrate satellite imagery for yield monitoring
- [ ] Add farmer feedback loop for model improvement

## 📄 Citation

If you use this work, please cite:

```
[Your Name]. (2026). Machine Learning-Based Crop Yield Prediction System 
for West Africa Under Climate Variability. 
GitHub repository: https://github.com/yourusername/crop-yield-prediction
```

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👥 Contributors

- [Your Name] - Initial work and development

## 🙏 Acknowledgments

- FAO for agricultural yield data
- World Bank for climate data
- NASA POWER for additional climate information
- scikit-learn community
- FastAPI framework

## 📧 Contact

For questions or collaboration:
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

## 🔗 Links

- [Documentation](docs/)
- [Dataset Information](data/README.md)
- [Model Details](ml_scripts/README.md)
