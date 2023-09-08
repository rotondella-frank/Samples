import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from fredapi import Fred

# API initialization
fred = Fred(api_key='24038268c5bd55ec155497b0079b5eb4')

# Retrieve data from FRED API
vix_data = pd.DataFrame(fred.get_series('VIXCLS', observation_start='2020-01-01', index_col=0))

# Dropping rows with null values
vix_data = vix_data.dropna()

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
vix_data_scaled = scaler.fit_transform(vix_data)

# Define a function to prepare the data for LSTM
def prepare_data(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

# Set the number of previous time steps to consider (look back)
look_back = 10

# Prepare the data
X, y = prepare_data(vix_data_scaled, look_back)

# Split the data into training and testing sets
train_size = int(0.998 * len(X))  # 99.8% of the data for training
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# Define and compile the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the LSTM model
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

# Make predictions on the test set
test_predictions = model.predict(X_test)

# Inverse transform the scaled predictions to original scale
test_predictions = scaler.inverse_transform(test_predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate the Mean Squared Error (MSE) for the test set
mse = mean_squared_error(y_test, test_predictions)
print(f"Mean Squared Error (MSE) on Test Set: {mse}")

# Number of additional days to forecast
forecast_days = 30

# Create an array to store forecasted values
forecasted_values = []

# Seed the model with the last 'look_back' values from the test set
seed_values = X_test[-1]

for _ in range(forecast_days):
    # Reshape the seed values to match the input shape of the model
    seed_values_reshaped = seed_values.reshape(1, look_back, 1)

    # Make a single-step prediction using the LSTM model
    next_value = model.predict(seed_values_reshaped)[0, 0]

    # Append the predicted value to the forecasted_values array
    forecasted_values.append(next_value)

    # Update the seed values for the next iteration (shift values and add the predicted value)
    seed_values = np.roll(seed_values, shift=-1)
    seed_values[-1] = next_value

# Inverse transform the scaled forecasted values to the original scale
forecasted_values = scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1))

# Generate dates for the forecasted period
forecast_start_date = vix_data.index[-1] + pd.DateOffset(days=1)
forecast_end_date = forecast_start_date + pd.DateOffset(days=forecast_days)
forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, closed='right')

# Create a DataFrame for the forecasted values with corresponding dates
forecast_df = pd.DataFrame({'Date': forecast_dates, 'VIX Forecast': forecasted_values[:, 0]})

# Combine the historical data and the forecast data for plotting
combined_data = pd.concat([vix_data, forecast_df])

# Plot the actual vs. predicted VIX values including the forecast
plt.figure(figsize=(10, 6))
plt.plot(combined_data.index, combined_data, label='Actual and Forecasted VIX', color='blue')
plt.legend()
plt.title('Actual and Forecasted VIX')
plt.xlabel('Date')
plt.ylabel('VIX')
plt.show()

# Print the forecasted values
print(forecast_df)
