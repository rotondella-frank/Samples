import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from fredapi import Fred

# API initialization
fred = Fred(api_key='24038268c5bd55ec155497b0079b5eb4')


# Retrieve data from FRED API
vix_data = pd.DataFrame(fred.get_series('VIXCLS', observation_start='2007-01-01', index_col=0))

# Dropping rows with null values
vix_data = vix_data.dropna()

# Split the data into training and testing sets
train_size = int(0.8 * len(vix_data))  # 80% of the data for training
train_vix, test_vix = vix_data[:train_size], vix_data[train_size:]

# Initialize lists to store benchmark metrics
p_values = []
d_values = []
q_values = []
mse_values = []

#Choose a forecasting method (ARIMA)
for p in range(1, 11):  # Trying p values from 1 to 10
    for d in range(1, 4):  # Trying d values from 1 to 3
        for q in range(1, 11):  # Trying q values from 1 to 10
            order = (p, d, q)
            
            try:
                # Train the ARIMA model
                model = ARIMA(train_vix, order=order)
                fitted_model = model.fit()

                # Make predictions
                predictions = fitted_model.forecast(steps=len(test_vix))

                # Evaluate the model
                mse = mean_squared_error(test_vix, predictions)
                mse_values.append(mse)

                # Store the hyperparameters for the best model (lowest MSE)
                p_values.append(p)
                d_values.append(d)
                q_values.append(q)

            except:
                # If the model fitting fails, continue to the next set of hyperparameters
                continue

# Find the best hyperparameters that give the lowest MSE
best_mse_idx = np.argmin(mse_values)
best_p = p_values[best_mse_idx]
best_d = d_values[best_mse_idx]
best_q = q_values[best_mse_idx]
best_mse = mse_values[best_mse_idx]

print(f"Best ARIMA Order: (p, d, q) = ({best_p}, {best_d}, {best_q})")
print(f"Best Mean Squared Error: {best_mse}")

# Optionally, you can plot the actual vs. predicted values of the best model to visualize the performance
best_model = ARIMA(train_vix, order=(best_p, best_d, best_q))
fitted_best_model = best_model.fit()
best_predictions = fitted_best_model.forecast(steps=len(test_vix))

plt.figure(figsize=(10, 6))
plt.plot(vix_data.index, vix_data, label='Actual VIX')
plt.plot(test_vix.index, best_predictions, label='Best Predicted VIX', color='red')
plt.legend()
plt.title('Actual vs. Best Predicted VIX')
plt.xlabel('Date')
plt.ylabel('VIX')
plt.show()
