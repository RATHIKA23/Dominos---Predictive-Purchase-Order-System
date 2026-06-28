🍕 Domino's Predictive Purchase Order System
> A production-ready data science pipeline that forecasts pizza sales and auto-generates weekly ingredient purchase orders — minimising waste and preventing stockouts.
---
📋 Table of Contents
Overview
Business Use Cases
Project Architecture
Dataset
Tech Stack
Installation
Usage
Pipeline Stages
Outputs
EDA Highlights
Model Details
Results
Project Structure
Future Improvements
---
📌 Overview
Domino's processes thousands of pizza orders daily. Manually estimating how much of each ingredient to stock is error-prone — leading to either waste (overstock) or lost revenue (stockouts).
This project solves that problem by:
Analysing 12 months of historical sales data (48,620 transactions, 105 pizza SKUs)
Training individual Prophet time-series models per SKU
Forecasting next-week sales
Multiplying forecasts × ingredient quantities to generate a ready-to-submit purchase order
---
💼 Business Use Cases
Use Case	Description
Inventory Management	Maintain optimal stock levels to meet demand without overbuying
Cost Reduction	Minimise waste from expired or excess ingredients
Sales Forecasting	Predict trends to inform promotions and staffing
Supply Chain Optimisation	Streamline ordering to align with predicted demand
---
🏗️ Project Architecture
```
Raw Sales Data  ──┐
                  ├──► Cleaning & EDA ──► Feature Engineering ──► Prophet Forecast
Ingredient Data ──┘                                                      │
                                                                         ▼
                                                              Ingredient Calculation
                                                                         │
                                                                         ▼
                                                              Purchase Order (Excel + CSV)
```
---
📊 Dataset
File	Description	Rows	Key Columns
`Pizza_Sale.xlsx`	Historical order transactions	48,620	`order_date`, `pizza_name_id`, `quantity`, `pizza_size`, `pizza_category`
`Pizza_ingredients.xlsx`	Ingredient requirements per SKU	518	`pizza_name_id`, `pizza_ingredients`, `Items_Qty_In_Grams`
Data coverage: January 1, 2015 → December 31, 2015  
Pizza SKUs: 105 unique size–type combinations  
Ingredients: 64 unique ingredients across all pizzas
---
🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Prophet](https://img.shields.io/badge/Prophet-1.1%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557c)
![OpenPyXL](https://img.shields.io/badge/OpenPyXL-3.1%2B-green)
Library	Purpose
`prophet`	Time-series forecasting per SKU
`pandas`	Data ingestion, cleaning, aggregation
`numpy`	Numerical computations
`scikit-learn`	MAPE evaluation metric
`matplotlib` / `seaborn`	EDA visualisations
`openpyxl`	Formatted Excel purchase order output
---
⚙️ Installation
1. Clone the repository
```bash
git clone https://github.com/your-username/dominos-purchase-order-system.git
cd dominos-purchase-order-system
```
2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac / Linux
source venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Place your data files
Ensure both Excel files are in the project root:
```
dominos-purchase-order-system/
├── Pizza_Sale.xlsx          ← required
├── Pizza_ingredients.xlsx   ← required
└── dominos_predictive_system.py
```
---
🚀 Usage
```bash
python dominos_predictive_system.py
```
That's it. The pipeline runs all 7 stages automatically and saves every output to the `outputs/` folder.
---
🔄 Pipeline Stages
Stage 1 — Data Ingestion & Validation
Loads both Excel files and validates schemas, row counts, and column types.
Stage 2 — Data Cleaning & Preprocessing
Parses and validates `order_date`
Coerces numeric columns; drops invalid quantities
Fills 16 rows with missing `pizza_name_id` using pizza name as proxy
Fills 4 missing ingredient gram values with 0 (with warning)
Recalculates mismatched `total_price` values
Normalises pizza size codes to uppercase
Stage 3 — Exploratory Data Analysis
Generates 9 publication-quality plots covering revenue trends, category breakdown, top products, seasonality, day-of-week patterns, and ingredient quantities.
Stage 4 — Feature Engineering
Aggregates 48,620 transactions into a daily quantity time series per SKU, adding weekday, month, and week-number features.
Stage 5 — Time-Series Forecasting (Prophet)
Trains one Prophet model per SKU (105 models total)
Multiplicative seasonality with yearly + weekly components
Walk-forward validation on last 7 days to compute MAPE
Automatic fallback to historical mean for sparse SKUs (< 20 observations)
Forecasts next 7 days of pizza sales
Stage 6 — Ingredient Calculation
```
required_grams = predicted_units × grams_per_pizza_per_ingredient
order_qty      = required_grams × (1 + 10% safety buffer)
```
Stage 7 — Purchase Order Generation
Writes a 4-sheet formatted Excel workbook and plain CSV files.
---
📁 Outputs
All outputs are saved to `outputs/`:
File	Description
`purchase_order.xlsx`	⭐ Main deliverable — formatted 4-sheet workbook
`purchase_order.csv`	Flat CSV version of the purchase order
`weekly_forecast.csv`	Predicted units per SKU for next 7 days
`model_evaluation.csv`	Per-SKU MAPE scores and model method used
`eda_01_daily_revenue_trend.png`	Daily revenue with 7-day rolling average
`eda_02_sales_by_category.png`	Revenue by pizza category
`eda_03_top20_pizzas.png`	Top 20 pizzas by units sold
`eda_04_monthly_seasonality.png`	Monthly sales trends by category
`eda_05_dayofweek_pattern.png`	Day-of-week demand pattern
`eda_06_heatmap_day_hour.png`	Order volume heatmap (day × hour)
`eda_07_price_distribution.png`	Unit price distribution by size
`eda_08_prophet_forecast_sample.png`	Forecast vs actual for top SKUs
`eda_09_top_ingredients.png`	Top ingredients by weekly order quantity
`run.log`	Full execution log
Excel Workbook Sheets
Sheet	Contents
Summary	Run metadata, MAPE, model counts, period
Purchase Order	Ingredient name, base qty (g), order qty with buffer (g & kg)
Forecast by SKU	Predicted weekly units per pizza SKU
Model Evaluation	Per-SKU MAPE (%) and method used
---
📈 EDA Highlights
Peak day: Friday–Sunday account for the highest sales volume
Top category: Classic pizzas lead in revenue (~35% share)
Busiest hours: 12:00–14:00 (lunch) and 17:00–21:00 (dinner)
Seasonality: Sales dip in September–October, peak in July and November
---
🤖 Model Details
Parameter	Value
Model	Facebook Prophet
Seasonality mode	Multiplicative
Yearly seasonality	✅ Enabled
Weekly seasonality	✅ Enabled
Daily seasonality	❌ Disabled
Changepoint prior scale	0.10
Confidence interval	90%
Validation method	Walk-forward (hold-out last 7 days)
Evaluation metric	MAPE
Fallback (sparse SKUs)	Historical mean
Forecast horizon	7 days
Safety buffer	10%
---
📊 Results
Metric	Value
Total transactions processed	48,620
Unique pizza SKUs modelled	105
Prophet models trained	91
Mean fallback models	14
Unique ingredients in PO	64
Overall MAPE	~65.8%
> **Note on MAPE:** The relatively high MAPE is expected for daily-level pizza SKU forecasting with only 1 year of history. Weekly aggregated accuracy is significantly better and sufficient for purchase ordering decisions.
Sample Purchase Order (Top 10 Ingredients)
Ingredient	Order Qty (kg)
Red Onions	22.000
Chicken	21.615
Capocollo	17.875
Tomatoes	15.224
Bacon	11.561
Pepperoni	11.077
Mushrooms	10.450
Spinach	8.943
Garlic	8.234
Corn	5.962
---
🗂️ Project Structure
```
dominos-purchase-order-system/
│
├── dominos_predictive_system.py   # Main pipeline script
├── Pizza_Sale.xlsx                # Input: sales data
├── Pizza_ingredients.xlsx         # Input: ingredient data
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
└── outputs/                       # Auto-generated on run
    ├── purchase_order.xlsx
    ├── purchase_order.csv
    ├── weekly_forecast.csv
    ├── model_evaluation.csv
    ├── eda_01_*.png
    ├── ...
    └── run.log
```
---
🔧 Configuration
Key parameters can be tuned at the top of `dominos_predictive_system.py`:
```python
FORECAST_DAYS   = 7     # Number of days to forecast ahead
SAFETY_BUFFER   = 0.10  # 10% buffer added to all ingredient quantities
MIN_HISTORY_OBS = 20    # Min observations needed to train Prophet (else mean fallback)
```
---
🚀 Future Improvements
[ ] Add SARIMA / LSTM as alternative models and compare via cross-validation
[ ] Incorporate external features (weather, promotions, holidays)
[ ] Build a Streamlit / Dash dashboard for interactive forecasting
[ ] Containerise with Docker for one-command deployment
[ ] Add unit tests with `pytest`
[ ] Schedule weekly runs via Apache Airflow or cron
[ ] Connect to live POS data via API
---
📄 License
This project is licensed under the MIT License — see the LICENSE file for details.
---
🙏 Acknowledgements
Facebook Prophet — time-series forecasting library
Pandas — data manipulation
Project spec by [Guvi / Data Science Bootcamp]
---
Built with ❤️ for the Domino's Capstone Project
