# TSA-Implement-Holt-Winters-method-in-Python
# Ex.No: 6 HOLT WINTERS METHOD
## Name: AKASH CT
## Date : 30/9/2025
## AIM:
To implement the Holt Winters Method Model using Python.
## ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative
trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and
Evaluate the model predictions against test data
6. Create teh final model and predict future data and plot it
## PROGRAM:
```
import pandas as pd
import numpy as np # Needed for np.sqrt (RMSE calculation)
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# --- 1. Load and Prepare Data ---
data = pd.read_csv('IMDB Top 250 Movies (1).csv', parse_dates=['year'], index_col=['year'])
data_series = data['rating']

# Resample to monthly frequency, fill missing with 0
data_monthly = data_series.resample('Y').sum().fillna(0)
print("--- Monthly Time Series Head ---")
print(data_monthly.head())

# --- 2. Scaling and Decomposition ---
# Plot raw data
plt.figure(figsize=(12, 4))
data_monthly.plot(title='Monthly Sum of Ratings (Raw Data)')
plt.xlabel('Date')
plt.ylabel('Sum of Ratings')
plt.grid(True)
plt.show()

# Scale the data
scaler = MinMaxScaler()
scaled_array = scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten()
scaled_data = pd.Series(scaled_array, index=data_monthly.index)

plt.figure(figsize=(12, 4))
scaled_data.plot(title='Monthly Sum of Ratings (Scaled Data)')
plt.xlabel('Date')
plt.ylabel('Scaled Value')
plt.grid(True)
plt.show()

# Decompose the scaled series (additive model)
decomposition = seasonal_decompose(scaled_data, model="additive", period=12)
fig = decomposition.plot()
fig.set_size_inches(10, 8)
plt.suptitle("Decomposed Plot (Scaled Data)", fontsize=14)
plt.show()

# Ensure positivity for Holt-Winters
if scaled_data.min() <= 0:
    scaled_data = scaled_data + 1

# --- 3. Train/Test Split and Model Evaluation ---
split_point = int(len(scaled_data) * 0.8)
train_data = scaled_data[:split_point]
test_data = scaled_data[split_point:]

print("\n--- Fitting Train Model (Add Trend, Mul Seasonality) ---")
model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

# Forecast test data
test_predictions_add = model_add.forecast(steps=len(test_data))

# Plot train, test, predictions
plt.figure(figsize=(12, 6))
ax = train_data.plot(label='Train Data')
test_predictions_add.plot(ax=ax, label='Test Predictions (Holt-Winters)', style='--')
test_data.plot(ax=ax, label='Test Data (Actual)')
ax.legend()
ax.set_title('Visual Evaluation: Holt-Winters Forecasting on Scaled Data')
plt.xlabel('Date')
plt.ylabel('Scaled Value')
plt.grid(True)
plt.show()

rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print(f"\nRoot Mean Squared Error (RMSE) on Test Set: {rmse:.4f}")
print(f"Standard Deviation of Scaled Data (Benchmark): {np.sqrt(scaled_data.var()):.4f}")
print(f"Mean of Scaled Data: {scaled_data.mean():.4f}")

# --- 4. Final Model and Forecast ---
print("\n--- Fitting Final Model on Full Data ---")

# Ensure positivity (shift by +1 if any zero or negative values exist)
final_series = data_monthly.copy()
if final_series.min() <= 0:
    final_series = final_series + 1

final_model = ExponentialSmoothing(
    final_series,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

forecast_steps = int(len(final_series) / 4)
final_predictions = final_model.forecast(steps=forecast_steps)

plt.figure(figsize=(12, 6))
ax = final_series.plot(label='Historical Monthly Sum of Ratings')
final_predictions.plot(ax=ax, label=f'Forecasted Sum of Ratings (Next {forecast_steps} Months)', style='--r')
ax.legend()
ax.set_title('Final Forecast: Holt-Winters Exponential Smoothing')
ax.set_xlabel('Date')
ax.set_ylabel('Sum of Ratings (shifted if necessary)')
plt.grid(True)
plt.show()

print(f"\n--- Final Forecast (Next {forecast_steps} Months) ---")
print(final_predictions)

```
## OUTPUT:
<img width="1324" height="665" alt="image" src="https://github.com/user-attachments/assets/c48d7299-1dbd-4e4a-8606-5af19d58a0c5" />
<img width="1390" height="509" alt="image" src="https://github.com/user-attachments/assets/e876c4dd-f8bf-4f80-a238-89e0313ee27a" />
<img width="1361" height="266" alt="image" src="https://github.com/user-attachments/assets/e86a0720-876c-4c87-aec0-409aca393d26" />
<img width="1374" height="751" alt="image" src="https://github.com/user-attachments/assets/527160cc-b0bc-4e6f-95aa-2e38ec08b094" />
<img width="1434" height="796" alt="image" src="https://github.com/user-attachments/assets/1dbeef3f-fb3a-4d92-b191-6861d9c2a760" />
<img width="1424" height="800" alt="image" src="https://github.com/user-attachments/assets/e5e80103-2fdf-4614-86bb-8467a8586b4d" />
<img width="430" height="650" alt="image" src="https://github.com/user-attachments/assets/b2f73945-eca9-46b6-95e2-d348ed87758b" />





## RESULT:
Thus the program run successfully based on the Holt Winters Method model.

