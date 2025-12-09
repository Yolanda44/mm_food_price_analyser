# Myanmar Food Price Intelligence
Myanmar food price analysis and interactive dashboard built to showcase applied data science skills for CS scholarship review. The project combines reproducible data cleaning, statistical analysis, forecasting, and geospatial storytelling to surface actionable insights from World Food Programme market monitoring data (2008-2025).

## Overview
- Dataset: 38,629 observations, 24 columns, 264 markets, 17 commodity families; coverage from 2008-01-15 to 2025-09-15.
- Source: WFP Myanmar market monitoring (raw file `wfp_food_prices_mmr.csv`), cleaned into `clean_wfp_food_prices.csv` with standardized commodity, unit, and admin names plus numeric conversions.
- Deliverables: Streamlit dashboard (`dashboard_app.py`), reproducible analysis script (`analysis.py`), exploratory notebooks, static figures (`figures/`), interactive market map (`maps/market_price_map.html`), and a narrative report (`report.md`).
- Tech stack: Python, pandas, NumPy, Plotly/Streamlit, seaborn/matplotlib, ARIMA forecasting, k-means clustering, z-score and percentage-change shock detection, and Folium/Mapbox for geospatial views.

## Objectives
- Build a transparent, reproducible pipeline from raw CSV to cleaned analytics-ready dataset.
- Quantify national inflation, regional disparities, and commodity-specific dynamics in Myanmar's food system.
- Detect shocks and volatility patterns linked to conflict, disasters, and currency shifts.
- Provide an interactive decision tool with forward-looking rice price forecasts and regional risk scoring.
- Demonstrate end-to-end engineering discipline (data QC, modular functions, caching, typed helpers, and clear documentation).

## Methods and Workflow
- Data cleaning: type coercion, text normalization, deduplication, unit standardization, and temporal feature engineering (`month`, `year`).
- Descriptive analytics: distribution plots, summary tables, commodity and market frequency counts.
- Time-series analysis: national trend lines, year-over-year change, staple price indexing by base year, and seasonal decomposition (e.g., onions).
- Regional lens: heatmaps, inequality box plots, pre/post-2021 comparisons, and Mapbox/Folium maps of market-level prices.
- Shock and volatility detection: coefficient of variation rankings, >=20% monthly jump flags by state/commodity, and z-score outlier detection for price spikes.
- Modeling: ARIMA-based 12-month rice price forecast with confidence intervals, k-means clustering of market trajectories, rice affordability index (rice price vs. wage), and composite regional risk scores combining volatility, inflation, and shock frequency.

## Key Insights 
- Long-run inflation: average prices rise steadily; largest month-on-month jumps occur in 2014-01 (+73.1%), 2011-06 (+68.9%), and 2011-07 (+53.1%).
- Regional inequality: Sagaing posts the highest average prices (~3,132 MMK), while Magway is lowest; inequality widens after 2021 with northern conflict-affected regions showing both higher means and wider spreads.
- Commodity dynamics: Pulses (+750%), tomatoes (+561%), and onions (+453%) are the most inflationary since their first observed year. Onions are also the most volatile commodity (CV ~1.30), while meat is the most stable (CV ~0.23).
- Seasonality: Onion prices peak mid-year and fall post-harvest, indicating clear supply-cycle effects.
- Currency divergence: MMK vs. USD prices correlate at 0.88, with widening gaps during MMK depreciation episodes.
- Shocks: 386 outlier records detected; shocks cluster around 2015 floods, COVID-19, and the 2021 crisis.
- Forecast: Rice prices are projected to remain elevated over the next 6-12 months (approx. 2,285-2,850 MMK range), staying above recent means; Mandalay shows a marked 2025 quake-related rice price jump.

## Dashboard Highlights (`dashboard_app.py`)
- Navigation-driven narrative: Overview -> National Trends -> Regional Story -> Commodity Deep Dive -> Rice Focus -> Shock & Volatility -> Future Outlook -> Final Summary.
- Hero metrics: overall inflation, shock counts, most volatile commodity, and peak volatility year.
- Interactive visuals: indexed staple trends, YoY changes, regional inequality plots, affordability index, Mandalay shock comparison, risk score leaderboard, and annotated event markers (2015 flood, 2020 COVID, 2021 crisis, 2025 earthquake).
- Performance: `st.cache_data` caching for fast reloads plus shared styling helpers for consistent look.

## Repository Guide
- `analysis.py`: End-to-end pipeline that cleans raw data, generates figures/maps, computes insights, and writes `report.md`.
- `analysis.ipynb`, `analysis_visualizations.ipynb`, `food_analysis.ipynb`, `food_analysis_v1.ipynb`: exploratory notebooks mirroring the scripted workflow.
- `dashboard_app.py`: Streamlit app for interactive exploration and storytelling.
- `figures/`: Saved PNG charts used in the report and dashboard.
- `maps/market_price_map.html`: Interactive Folium map of markets and average prices.
- `clean_wfp_food_prices.csv`: Cleaned dataset consumed by the dashboard; `wfp_food_prices_mmr.csv` is the raw source.
- `requirements.txt`: Minimal runtime dependencies for the dashboard.

## How to Run
1) Python environment (3.10+ recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```
2) Launch the dashboard:
   ```bash
   streamlit run dashboard_app.py
   ```
3) Reproduce figures and report from raw data (optional, requires a few extra packages beyond `requirements.txt`):
   ```bash
   python -m pip install matplotlib seaborn folium scikit-learn statsmodels
   python analysis.py
   ```
   Outputs: refreshed `report.md`, plots in `figures/`, and `maps/market_price_map.html`.

## Impact and Future Work
- Policy relevance: pinpoints supply-chain bottlenecks (onions, oils), highlights high-risk regions for pre-positioning aid, and provides affordability and risk indicators for transfer value calibration.
- Engineering extensions: automate data refresh from WFP feeds, add CI to lint/test data transformations, integrate FX feeds for real-time divergence monitoring, and experiment with gradient boosting or Prophet models for improved forecasts.
- Research directions: couple rainfall or conflict data for causal inference, test price pass-through elasticity across regions, and build an alerting pipeline around the shock detectors and affordability index.

## Summary
An end-to-end, reproducible data science system that turns 17 years of Myanmar food market data into interactive intelligence - combining rigorous cleaning, statistical modeling, forecasting, geospatial analysis, and clear storytelling to support real-world food security decisions.
