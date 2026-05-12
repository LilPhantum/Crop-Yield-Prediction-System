"""
=================================================================================
PRODUCTION DEPLOYMENT & 2026 SCENARIO PREDICTIONS
Ultimate Crop Yield Forecasting System
=================================================================================

This module provides:
1. 2026 baseline yield predictions
2. Climate scenario analysis
3. Agricultural recommendations
4. Backend API integration

Author: Advanced ML System
Date: 2026
=================================================================================
"""

import pandas as pd
import numpy as np
import joblib
import os


class ProductionPredictor:
    """
    Production deployment for crop yield predictions.
    """

    def __init__(self, models_dir="backend/models", data_path=None):
        self.models_dir = models_dir
        self.data_path = data_path
        self.loaded_models = {}

    def load_model(self, crop_name):
        """
        Load trained model for a crop.
        """

        crop_key = crop_name.replace(" ", "_").replace(",", "")
        model_path = os.path.join(
            self.models_dir,
            f"model_{crop_key}_ultimate.pkl"
        )

        if crop_key in self.loaded_models:
            return self.loaded_models[crop_key]

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found for {crop_name}: {model_path}"
            )

        model_artifact = joblib.load(model_path)
        self.loaded_models[crop_key] = model_artifact

        print(
            f"✓ Loaded {crop_name} model "
            f"(R² = {model_artifact['metrics']['test_r2']:.4f})"
        )

        return model_artifact

    def prepare_prediction_features(
        self,
        rainfall,
        temperature,
        pesticides,
        year=2026
    ):
        """
        Create feature values for prediction.
        """

        features = {}

        features["rainfall"] = rainfall
        features["temperature"] = temperature
        features["pesticides"] = pesticides

        features["temp_rain_interaction"] = temperature * rainfall / 1000
        features["pesticide_intensity"] = pesticides / (rainfall + 1)

        features["growing_degree_days"] = max(temperature - 10, 0) * 365
        features["heat_stress"] = max(temperature - 30, 0)
        features["cold_stress"] = max(10 - temperature, 0)

        features["temp_deviation_from_25"] = abs(temperature - 25)

        features["temperature_sq"] = temperature ** 2
        features["rainfall_sq"] = rainfall ** 2
        features["pesticides_sq"] = pesticides ** 2

        features["log_pesticides"] = np.log1p(pesticides)
        features["log_rainfall"] = np.log1p(rainfall)

        features["years_since_1990"] = year - 1990
        features["tech_trend"] = (year - 1990) ** 0.5

        if rainfall < 500:
            features["rain_cat_dry"] = 1
            features["rain_cat_moderate"] = 0
            features["rain_cat_wet"] = 0
            features["rain_cat_very_wet"] = 0

        elif rainfall < 1000:
            features["rain_cat_dry"] = 0
            features["rain_cat_moderate"] = 1
            features["rain_cat_wet"] = 0
            features["rain_cat_very_wet"] = 0

        elif rainfall < 1500:
            features["rain_cat_dry"] = 0
            features["rain_cat_moderate"] = 0
            features["rain_cat_wet"] = 1
            features["rain_cat_very_wet"] = 0

        else:
            features["rain_cat_dry"] = 0
            features["rain_cat_moderate"] = 0
            features["rain_cat_wet"] = 0
            features["rain_cat_very_wet"] = 1

        return features

    def predict(self, crop_name, rainfall, temperature, pesticides, year=2026):
        """
        Make yield prediction.

        Returns:
            Predicted yield in hg/ha.
        """

        model_artifact = self.load_model(crop_name)

        features_dict = self.prepare_prediction_features(
            rainfall,
            temperature,
            pesticides,
            year
        )

        feature_cols = list(model_artifact["features"])

        feature_frame = pd.DataFrame(
            [[features_dict.get(feature, 0) for feature in feature_cols]],
            columns=feature_cols
        )

        if model_artifact["type"] == "Ensemble":
            rf = model_artifact["rf_model"]
            gb = model_artifact["gb_model"]
            ridge = model_artifact["ridge_model"]
            scaler = model_artifact["ridge_scaler"]
            weights = model_artifact["weights"]

            pred_rf = rf.predict(feature_frame)[0]
            pred_gb = gb.predict(feature_frame)[0]

            feature_frame_scaled = scaler.transform(feature_frame)
            pred_ridge = ridge.predict(feature_frame_scaled)[0]

            prediction = (
                weights["rf"] * pred_rf +
                weights["gb"] * pred_gb +
                weights["ridge"] * pred_ridge
            )

        elif model_artifact["type"] == "Ridge":
            scaler = model_artifact["model"]["scaler"]
            model = model_artifact["model"]["model"]

            feature_frame_scaled = scaler.transform(feature_frame)
            prediction = model.predict(feature_frame_scaled)[0]

        else:
            model = model_artifact["model"]
            prediction = model.predict(feature_frame)[0]

        return prediction

    def get_baseline_2026_conditions(self, region="West Africa"):
        """
        Get baseline climate and agricultural conditions for 2026.
        """

        baselines = {
            "West Africa": {
                "rainfall": 1200,
                "temperature": 26.5,
                "pesticides": 5000
            },
            "East Africa": {
                "rainfall": 950,
                "temperature": 22.5,
                "pesticides": 4500
            },
            "Southern Africa": {
                "rainfall": 650,
                "temperature": 20.5,
                "pesticides": 8000
            }
        }

        return baselines.get(region, baselines["West Africa"])

    def run_scenario_analysis(self, crop_name, region="West Africa"):
        """
        Run complete scenario analysis for 2026.
        """

        print("\n" + "=" * 80)
        print(f" 2026 SCENARIO ANALYSIS: {crop_name}")
        print("=" * 80)

        baseline = self.get_baseline_2026_conditions(region)

        print(f"\nBaseline 2026 Conditions ({region}):")
        print(f"  Rainfall:     {baseline['rainfall']:,} mm/year")
        print(f"  Temperature:  {baseline['temperature']:.1f}°C")
        print(f"  Pesticides:   {baseline['pesticides']:,} tonnes")

        scenarios = {
            "Baseline (2026 Trends)": {
                "rainfall": baseline["rainfall"],
                "temperature": baseline["temperature"],
                "pesticides": baseline["pesticides"],
                "description": "Current trends continue"
            },

            "Optimistic (Good Practices)": {
                "rainfall": baseline["rainfall"],
                "temperature": baseline["temperature"],
                "pesticides": baseline["pesticides"] * 1.3,
                "description": "Improved input usage"
            },

            "Moderate Warming (+1°C)": {
                "rainfall": baseline["rainfall"],
                "temperature": baseline["temperature"] + 1.0,
                "pesticides": baseline["pesticides"],
                "description": "Temperature increase of 1°C"
            },

            "High Warming (+2°C)": {
                "rainfall": baseline["rainfall"],
                "temperature": baseline["temperature"] + 2.0,
                "pesticides": baseline["pesticides"],
                "description": "Temperature increase of 2°C"
            },

            "Severe Warming (+3°C)": {
                "rainfall": baseline["rainfall"],
                "temperature": baseline["temperature"] + 3.0,
                "pesticides": baseline["pesticides"],
                "description": "Temperature increase of 3°C"
            },

            "Moderate Drought (-15% rain)": {
                "rainfall": baseline["rainfall"] * 0.85,
                "temperature": baseline["temperature"],
                "pesticides": baseline["pesticides"],
                "description": "Rainfall reduction of 15%"
            },

            "Severe Drought (-30% rain)": {
                "rainfall": baseline["rainfall"] * 0.70,
                "temperature": baseline["temperature"],
                "pesticides": baseline["pesticides"],
                "description": "Rainfall reduction of 30%"
            },

            "Wet Conditions (+20% rain)": {
                "rainfall": baseline["rainfall"] * 1.20,
                "temperature": baseline["temperature"],
                "pesticides": baseline["pesticides"],
                "description": "Rainfall increase of 20%"
            },

            "Hot & Dry (+2°C, -20% rain)": {
                "rainfall": baseline["rainfall"] * 0.80,
                "temperature": baseline["temperature"] + 2.0,
                "pesticides": baseline["pesticides"],
                "description": "Combined heat and rainfall stress"
            },

            "Extreme Stress (+3°C, -30% rain)": {
                "rainfall": baseline["rainfall"] * 0.70,
                "temperature": baseline["temperature"] + 3.0,
                "pesticides": baseline["pesticides"],
                "description": "Severe climate stress"
            },

            "Best Management (+inputs, +rain)": {
                "rainfall": baseline["rainfall"] * 1.10,
                "temperature": baseline["temperature"],
                "pesticides": baseline["pesticides"] * 1.5,
                "description": "Improved input usage and favourable rainfall"
            }
        }

        results = {}

        print("\n" + "-" * 80)
        print(f"{'Scenario':<40} {'Yield (hg/ha)':>15} {'Change':>15}")
        print("-" * 80)

        for scenario_name, conditions in scenarios.items():
            prediction = self.predict(
                crop_name,
                conditions["rainfall"],
                conditions["temperature"],
                conditions["pesticides"]
            )

            results[scenario_name] = {
                "yield": prediction,
                "conditions": conditions,
                "description": conditions["description"]
            }

        baseline_yield = results["Baseline (2026 Trends)"]["yield"]

        for scenario_name, data in results.items():
            yield_val = data["yield"]

            if scenario_name == "Baseline (2026 Trends)":
                change_str = "---"
            else:
                change = yield_val - baseline_yield
                change_pct = (change / baseline_yield) * 100
                change_str = f"{change:>+8,.0f} ({change_pct:>+5.1f}%)"

            print(f"{scenario_name:<40} {yield_val:>15,.0f} {change_str:>15}")

        print("-" * 80)

        all_yields = [data["yield"] for data in results.values()]
        best_yield = max(all_yields)
        worst_yield = min(all_yields)
        uncertainty_range = best_yield - worst_yield

        print("\nRisk Assessment:")
        print(f"  Baseline (2026):     {baseline_yield:>12,.0f} hg/ha")
        print(
            f"  Best case:           {best_yield:>12,.0f} hg/ha "
            f"(+{((best_yield / baseline_yield - 1) * 100):>5.1f}%)"
        )
        print(
            f"  Worst case:          {worst_yield:>12,.0f} hg/ha "
            f"({((worst_yield / baseline_yield - 1) * 100):>+6.1f}%)"
        )
        print(f"  Uncertainty range:   {uncertainty_range:>12,.0f} hg/ha")

        recommendation = self.generate_recommendation(
            crop_name,
            results,
            baseline_yield
        )

        print("\n" + "=" * 80)
        print(" AI RECOMMENDATION")
        print("=" * 80)
        print(recommendation)

        return results

    def generate_recommendation(self, crop_name, scenario_results, baseline_yield):
        """
        Generate crop recommendation from scenario results.
        """

        hot_dry_yield = scenario_results["Hot & Dry (+2°C, -20% rain)"]["yield"]
        extreme_yield = scenario_results["Extreme Stress (+3°C, -30% rain)"]["yield"]
        best_mgmt_yield = scenario_results["Best Management (+inputs, +rain)"]["yield"]

        hot_dry_impact = ((hot_dry_yield - baseline_yield) / baseline_yield) * 100
        extreme_impact = ((extreme_yield - baseline_yield) / baseline_yield) * 100
        mgmt_impact = ((best_mgmt_yield - baseline_yield) / baseline_yield) * 100

        crop_key = crop_name.replace(" ", "_").replace(",", "")
        model_r2 = self.loaded_models[crop_key]["metrics"]["test_r2"]

        recommendation = f"""
📊 CROP: {crop_name}
📅 PROJECTION YEAR: 2026

🎯 BASELINE PROJECTION:
   Expected yield under current trends: {baseline_yield:,.0f} hg/ha

⚠️ CLIMATE RISK ANALYSIS:
   • Hot and dry scenario:     {hot_dry_impact:+.1f}% yield change
   • Extreme stress:           {extreme_impact:+.1f}% yield change
   • Best management:          {mgmt_impact:+.1f}% yield change

💡 KEY FINDINGS:
   The model shows that climate conditions and agricultural input usage
   both influence predicted crop yield.

📋 RECOMMENDATIONS:

"""

        if extreme_impact < -15:
            recommendation += """   🔴 HIGH RISK UNDER CLIMATE STRESS

   Critical Actions:
   1. Use drought tolerant crop varieties where possible
   2. Implement water conservation practices
   3. Plan for supplemental irrigation if available
   4. Diversify crops to reduce production risk
   5. Improve farm input management during favourable seasons

"""

        elif hot_dry_impact < -8:
            recommendation += """   🟡 MODERATE CLIMATE RISK

   Recommended Actions:
   1. Monitor weather forecasts closely
   2. Optimize planting dates for expected conditions
   3. Improve input planning before planting
   4. Consider climate adapted crop varieties
   5. Implement soil moisture conservation

"""

        else:
            recommendation += """   🟢 LOW TO MODERATE CLIMATE RISK

   Standard Best Practices:
   1. Maintain good agricultural practices
   2. Use recommended input levels
   3. Implement integrated pest management
   4. Time field operations properly
   5. Maintain soil health

"""

        recommendation += f"""
✅ SPECIFIC ADVICE FOR {crop_name.upper()}:

   • Optimal temperature range: 22 to 28°C
   • Water requirements: {1200 if crop_name == "Rice, paddy" else 800} to 1500 mm
   • Critical growth periods: Monitor moisture during flowering and grain filling
   • Best management scenario shows {mgmt_impact:.1f}% yield change

📊 DATA CONFIDENCE:
   Model accuracy (R²): {model_r2:.2f}
   This represents {"strong" if model_r2 > 0.70 else "moderate"} predictive confidence.

⚡ CONCLUSION:
   The prediction result shows the importance of combining climate awareness
   with improved agricultural management.
"""

        return recommendation


def run_production_deployment():
    """
    Complete production deployment with scenario analysis.
    """

    print("\n" + "=" * 80)
    print(" PRODUCTION DEPLOYMENT - 2026 PREDICTIONS")
    print(" Ultimate Crop Yield Forecasting System")
    print("=" * 80)

    predictor = ProductionPredictor()

    crops = [
        "Maize",
        "Rice, paddy",
        "Wheat",
        "Cassava",
        "Sorghum",
        "Potatoes",
        "Soybeans"
    ]

    print("\nRunning scenario analysis for major crops...")
    print("This will generate 2026 predictions under 11 different scenarios.")

    all_results = {}

    for crop in crops:
        try:
            results = predictor.run_scenario_analysis(
                crop,
                region="West Africa"
            )
            all_results[crop] = results

        except FileNotFoundError:
            print(f"\n⚠ Model not found for {crop}. Skipping.")

        except Exception as e:
            print(f"\n✗ Error analyzing {crop}: {e}")

    print("\n" + "=" * 80)
    print(" DEPLOYMENT COMPLETE")
    print("=" * 80)
    print(f"\n✓ Analyzed {len(all_results)} crops")
    print("✓ Generated 2026 baseline predictions")
    print("✓ Evaluated 11 climate and management scenarios per crop")
    print("✓ Produced recommendations")

    print("\n✓ Ready for backend integration")
    print("✓ Models can now be called through the API for real time predictions")

    return all_results


if __name__ == "__main__":
    results = run_production_deployment()