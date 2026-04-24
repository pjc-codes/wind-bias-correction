# Wind Forecast Bias Correction Model

This project corrects systematic wind forecast bias by learning a mapping from Open-Meteo historical forecast data to ERA5 historical weather reanalysis (ground truth). 

We choose 10 locations across multiple terrains (in order to maximize the ), conduct feature engineering, a train/test split across 2025, train a correction model, and evaluate on test set (nov+dec). 

We build baseline models using Ridge and Lasso linear regression models. We compare with LightGBM model. Tree-based ensemble model is a good choice because of the fact that they tend to work well for such varied feature sets and this volume of data as well as the fact that they can capture non-linear interactions better than standard OLS regression. 

We see that the best performance was for LightGBM model. And we conduct further analysis using that model. We evaluate the model on the test set (Nov and Dec data) and calculate metrics such as MSE, MAE and bias (mean of residuals).

## Files
- `get_data.ipynb` : notebook for using the API to fetch the wind data.
- `wind-pred.ipynb`: end-to-end notebook for preprocessing data, training, evaluation, and analysis.
- `predictions.csv`: held-out predictions for every station-hour in November and December 2025.
- `figs/`: terrain, temporal, and raw-vs-corrected comparison plots and tables.
- `data/`: all the data acquired and processed in the analysis.
- `x_model.pkl`,`y_model.pkl` : trained model artifacts predicting the x and y speed components.

## Location Selection

All 10 locations are in India so the full study stays on a single local timezone while covering multiple terrain regimes. 

| Terrain  | Locations           |
| -------- | ------------------- |
| Coastal  | Mumbai, Chennai     |
| Mountain | Shimla, Leh         |
| Plains   | New Delhi, Lucknow  |
| Tropical | Agartala, Bengaluru |
| Desert   | Bikaner, Jaisalmer  |

We have chosen two locations from each terrain as this would be an equitable way to gather data. However, if we were to fetch more data from more locations for each terrain, we would expect better generalization than at present.

## Data And Split

- Forecast input: Open-Meteo Historical Forecast API
- Ground truth: Open-Meteo Historical Weather API with `models=era5`
- API code directly taken from the Open-Meteo website.
- Variables: hourly `wind_speed_10m` and `wind_direction_10m`
- Date range: `2025-01-01` through `2025-12-31` (entire calendar year of 2025).
- Consolidation of datasets is done through merging and concatenation.
- Training set: January through October 2025
- Held-out test set: November and December 2025

The final `predictions.csv` contains `14,640` held-out rows ie `10 locations x 61 days x 24 hours`.

## Modeling Approach

I modeled wind using **x-y vector components** rather than raw speed and direction. That avoids circular-angle issues and makes the learning target smoother. The model itself is a pair of `HistGradientBoostingRegressor` estimators, one for `x` and one for `y`.

Features include:

- Raw forecast speed and direction
- Forecast-derived `x-y` components and direction sin/cos terms
- Location latitude, longitude, and elevation
- Terrain one-hot features
- Month, hour, day-of-year, and cyclic sin/cos time features
- Season one-hot features

This is a good fit for the problem because forecast bias is definitely non-linear and depends on terrain, season, and time of day.

Gradient boosting models (like LightGBM, XGBoost, HistGradientBoosting) etc capture these interactions well without heavy feature scaling, and they handle mixed feature types effectively as well. Moreover, gradient boosting models explicitly work to fit residuals sequentially, which maps exactly to what we are attempting here (ie bias correction).

## Held-Out Test Results

We see that we get the following improvements in the test set:

- Wind speed bias improved by `64%`
- Wind direction (MAE) improved by `14%`

Thus, our correction model is certainly able to capture effects better than the raw forecast model did.

## Analysis

### 1. Terrain dependence

We see that the forecast bias depends strongly on terrain. We can see the evidence in the plots where we have visualized prediction bias against different terrains (in `figs/`)

What we notice is that:

- Mountains are qualitatively different from the rest: the forecast over-predicts there, while other terrains are under-predicted.
- Desert sites have the largest bias, which should be the consequence of local high-heat currents and other such effects.
- The correction model helps everywhere, but the gain is largest in mountains, suggesting that terrain-induced bias is learnable from context features.

### 2. Temporal dependence

The bias also depends on time of day and season. This can be seen from the plots very evidently (see saved visualizations in `figs`). See also the calculated tables with initial metrics and final metrics and the % improvements.

We see this both in the case of hour of day as well as seasons. Insights:

- The forecast has a stronger negative bias overnight and in the morning than in mid-afternoon.
- Even as the correction model reduces bias in every season,more features would help further generalize.

### Leave-One-Location-Out (LOLO) Validation 

We conduct LOLO validation to see how well the model generalizes to unseen locations. Unsurprisingly, they do not generalize too well.

We need more granular terrain data to capture the terrain effects better.


## How to Improve The Model 

- Use multiyear data to train the model. At the moment we are only using a single calendar year.
- Perform hyperparameter tuning to extract maximum generalization from the LightGBM correction model.
- Add lagged features (t-1, t-3, t-6 hours) and rolling means. 
- Use higher-resolution terrain descriptor-labels instead of the broad category we have now.
- Add more comparitive analysis of terrain, season etc.
- Conduct feature importance analysis for better model interpretability/explainability.
- Try different models per terrain and see if that leads to better bias correction.

