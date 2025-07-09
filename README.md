# What Drives the Price of a Used Car?

This project analyzes over 400,000 used car listings to identify what factors influence a car's price. I built this for a dealership that wants to fine-tune its inventory and pricing strategy based on real-world data. The project follows the CRISP-DM process and is structured for both technical and non-technical audiences.

## 🧠 Business Objective

- Understand which vehicle attributes increase or decrease resale value
- Guide dealership decisions for pricing and inventory
- Present insights clearly for business use i.e. which brands are overpriced or undervalued

## 📁 Dataset Overview

- Original source: Kaggle (3M+ listings)
- Filtered version used: ~426,000 listings
- After cleaning: ~93,000 dealership-relevant listings
- Includes fields like:
  - Make and model
  - Year, mileage, fuel type, condition
  - Title status, transmission, drive type
  - Region and color

## 🛠️ Data Cleaning and Preparation

- Removed listings with invalid VINs, duplicate entries, or missing critical data
- Filtered out junk i.e. cars listed for under $1,000 or with salvage titles
- Standardized inconsistent model names (e.g. “F 150” to “F-150”)
- Created:
  - `car_age` (based on listing year)
  - `log_price` (to normalize skewed price distribution)
  - `region_grouped` (to reduce high-cardinality noise)
- One-hot encoded all categorical variables for modeling

## 📊 Modeling Process

- Split data into training and test sets (80/20)
- Used pipelines with `ColumnTransformer` for scaling and encoding
- Trained 3 models:
  - Linear Regression
  - Ridge Regression (best performer)
  - Lasso Regression (feature shrinkage)
- Tuned with GridSearchCV and LassoCV
- Evaluated using:
  - RMSE
  - R²
  - 5-fold cross-validated R²

### 📈 Performance Summary

| Model              | R² Score | CV R² (5-fold) | RMSE  |
|--------------------|----------|----------------|-------|
| Ridge Regression   | 0.669    | 0.460          | 0.435 |
| Linear Regression  | 0.669    | 0.460          | 0.435 |
| Lasso Regression   | 0.669    | 0.458          | 0.435 |

- Ridge was selected due to better generalization and handling of one-hot encoded features

## 🔍 Key Findings

- **Top positive price drivers:**
  - Ferrari, Porsche, Tesla
  - Low mileage and newer year
  - Clean title and excellent condition
- **Top negative price drivers:**
  - Models like PT Cruiser, Saturn, and Cobalt
  - Poor condition or excessive mileage
  - Older age or uncommon cylinder types
- **Depreciation insights:**
  - Estimated ~36% price drop per year of age
  - Estimated ~17% depreciation per odometer log unit
- Used grouped bar plots to visualize:
  - Top and bottom feature impacts by category
  - Manufacturer and model trends
  - Drive type, fuel type, and transmission effects

## 📂 Repo Contents

- `usedCarAnalysis.ipynb` – Main notebook with step-by-step analysis
- `model_coefficients_comparison.csv` – Feature weights for each model
- `model_results_summary.csv` – Evaluation metrics comparison
- `vehicles.csv` the dataset used for training the model
- Visuals and grouped plots inline

## 🚙 Who This Helps

- Dealerships pricing used cars for retail
- Data-driven sales managers and inventory planners
- Analysts trying to predict market pricing behavior

## 🧪 Price Predictor Tool

I also included a basic price prediction program inside the notebook. After training the Ridge model, the pipeline is saved and can be reused to estimate log-price for any new listing, which can then be converted back to dollar price.

You can test custom input data i.e. a 2018 Toyota Camry with 50,000 miles and clean condition, and the model will return a price estimate based on learned patterns.

This can help dealerships:
- Run bulk price forecasts
- Evaluate trade-ins quickly
- Make smarter purchase decisions at auctions
