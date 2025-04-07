import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from fredapi import Fred
import matplotlib.image as mpimg
import matplotlib.image as mpimg
import matplotlib.dates as mdates

# Load the CSV file for daily stock price data
price_csv_path = '~/Downloads/HII (3).csv'
brk_b_df = pd.read_csv(price_csv_path, parse_dates=['Date'], index_col='Date')

# Resample the Adj Close column to quarterly data
brk_b_df_quarterly = brk_b_df['Adj Close'].resample('Q').mean()

# Load the CSV file into a DataFrame for profitloss data (quarterly)
csv_path = '~/Downloads/HII_quarterly_financials (2).csv'
pl_hist_df = pd.read_csv(csv_path).apply(lambda x: x.str.strip().replace({'\$': '', ',': ''}, regex=True) if x.dtype == 'object' else x)

# Filter rows based on 'name' column values and set the 'name' column as the index
pl_hist_df = pl_hist_df[pl_hist_df['name'].isin(['TotalRevenue', 'NetIncome', 'DilutedAverageShares', 'DilutedEPS', 'InterestExpense'])]
pl_hist_df.set_index('name', inplace=True)
pl_hist_df = pl_hist_df.drop('ttm', axis=1)

# Transpose the profit/loss data to have 'Date' as rows and convert the index to datetime
pl_hist_df = pl_hist_df.T
pl_hist_df.index = pd.to_datetime(pl_hist_df.index)
pl_hist_df = pl_hist_df.sort_index()  # Ensure the index is sorted to be monotonic

# Reindex to ensure all quarters are included
all_quarters = pd.date_range(start=pl_hist_df.index.min(), end=pl_hist_df.index.max(), freq='Q')
pl_hist_df = pl_hist_df.reindex(all_quarters)

# Convert the data columns to numeric (if not already)
pl_hist_df['TotalRevenue'] = pd.to_numeric(pl_hist_df['TotalRevenue'], errors='coerce')
pl_hist_df['NetIncome'] = pd.to_numeric(pl_hist_df['NetIncome'], errors='coerce').ffill()
pl_hist_df['DilutedAverageShares'] = pd.to_numeric(pl_hist_df['DilutedAverageShares'], errors='coerce').ffill()
pl_hist_df['DilutedEPS'] = pd.to_numeric(pl_hist_df['DilutedEPS'], errors='coerce').ffill()

# Calculate the Net Profit Margin
pl_hist_df['NetProfitMargin'] = pl_hist_df['NetIncome'] / pl_hist_df['TotalRevenue']

# Drop any NaN values to avoid issues with forecasting
pl_hist_df.fillna(0, inplace=True)

# Load balance sheet data to calculate WACC
balance_sheet_csv_path = '~/Downloads/HII_quarterly_balance-sheet (1).csv'
balance_sheet_df = pd.read_csv(balance_sheet_csv_path).apply(lambda x: x.str.strip().replace({'\$': '', ',': ''}, regex=True) if x.dtype == 'object' else x)

# Filter relevant rows for debt and equity and set 'name' as the index
balance_sheet_df = balance_sheet_df[balance_sheet_df['name'].isin(['TotalDebt', 'TotalEquityGrossMinorityInterest','TotalAssets','TotalLiabilitiesNetMinorityInterest'])]
balance_sheet_df.set_index('name', inplace=True)

# Transpose and convert index to datetime
balance_sheet_df = balance_sheet_df.T
balance_sheet_df.index = pd.to_datetime(balance_sheet_df.index)

# Reindex to ensure all quarters are included
all_quarters = pd.date_range(start=balance_sheet_df.index.min(), end=balance_sheet_df.index.max(), freq='Q')
balance_sheet_df = balance_sheet_df.reindex(all_quarters).fillna(0)

# Convert data to numeric
balance_sheet_df['TotalDebt'] = pd.to_numeric(balance_sheet_df['TotalDebt'], errors='coerce').ffill()
balance_sheet_df['TotalAssets'] = pd.to_numeric(balance_sheet_df['TotalAssets'], errors='coerce').ffill()
balance_sheet_df['TotalEquityGrossMinorityInterest'] = pd.to_numeric(balance_sheet_df['TotalEquityGrossMinorityInterest'], errors='coerce').ffill()
balance_sheet_df['TotalLiabilitiesNetMinorityInterest'] = pd.to_numeric(balance_sheet_df['TotalLiabilitiesNetMinorityInterest'], errors='coerce').ffill()



# Drop any NaN values to avoid issues
balance_sheet_df.fillna(0, inplace=True)

# Fetch 5-year rate data from FRED
fred = Fred(api_key='24038268c5bd55ec155497b0079b5eb4')
rates = fred.get_series('DGS5', start_date='1985-01-01', end_date='2024-12-31')
rates = rates.resample('Q').ffill() / 100  # Resample to quarterly and use the latest 5-year rates
market = fred.get_series('sp500', start_date='1985-01-01', end_date='2024-12-31')
market = market.resample('Q').ffill() / 100  # Resample to quarterly and use the latest 5-year rates

# CAPM Calculation
expected_market_return = 0.10  # Assumed expected market return (can be adjusted)
beta =.50 # Assumed beta value for the company (can be adjusted)

# Calculate NPV with correct alignment to historical indices
historical_npvs = []

# Loop through each historical period to calculate NPV
for i in range(len(pl_hist_df)):
    current_date = pl_hist_df.index[i]

    # Set the starting point for the forecast
    forecast_start_date = current_date + pd.DateOffset(months=3)
    forecast_dates = pd.date_range(start=forecast_start_date, periods=20, freq='Q')

    # Forecast diluted average shares for each iteration
    shares_series = pl_hist_df['DilutedAverageShares'][:i+1].dropna().astype(float)
    if len(shares_series) < 2:
        continue
    elif len(shares_series) < 8:
        shares_forecast = SimpleExpSmoothing(shares_series).fit().forecast(steps=20)
    else:
        shares_forecast = ExponentialSmoothing(shares_series, trend='add', seasonal='add', seasonal_periods=4).fit().forecast(steps=20)

    # Calculate the risk-free rate for the current quarter
    risk_free_rate = rates.loc[:current_date].iloc[-1]

    # Calculate cost of equity for the current quarter
    cost_of_equity = risk_free_rate + beta * (expected_market_return - risk_free_rate)

    # Get debt and equity values for the current quarter
    total_debt = balance_sheet_df['TotalDebt'].loc[:current_date].iloc[-1] if not balance_sheet_df['TotalDebt'].loc[:current_date].empty else None
    total_equity = balance_sheet_df['TotalEquityGrossMinorityInterest'].loc[:current_date].iloc[-1] if not balance_sheet_df['TotalEquityGrossMinorityInterest'].loc[:current_date].empty else None

    # Skip iteration if debt or equity data is missing
    if total_debt is None or total_equity is None:
        continue

    # Get interest expense for the current quarter
    interest_expense = pd.to_numeric(pl_hist_df['InterestExpense'].iloc[i], errors='coerce')

    # Cost of Debt for the current quarter
    if total_debt == 0:
        cost_of_debt = 0.05
    else:
        cost_of_debt = interest_expense / total_debt

    # Calculate Debt and Equity Weights for the current quarter
    total_value = total_debt + total_equity
    if total_value == 0:
        debt_weight = 0
        equity_weight = 0
    else:
        debt_weight = total_debt / total_value
        equity_weight = total_equity / total_value

    # Tax Rate Assumption for WACC Calculation
    tax_rate = 0.25

    # Calculate WACC for the current quarter
    wacc =(equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))
    quarterly_wacc = (1 + wacc) ** (1/4) - 1
    quarterly_wacc = max(quarterly_wacc, 0.0001)

    # Update discount factors based on the current historical index
    discount_factors = [(1 / (1 + quarterly_wacc)) ** j for j in range(1, 21)]

    # Forecast revenues and net profit margins
    if len(pl_hist_df['TotalRevenue'][:i+1]) < 8:
        revenue_forecast = SimpleExpSmoothing(pl_hist_df['TotalRevenue'][:i+1].ffill()).fit().forecast(steps=20)
    else:
        revenue_forecast = ExponentialSmoothing(pl_hist_df['TotalRevenue'][:i+1].ffill(), trend='add', seasonal='add', seasonal_periods=4).fit().forecast(steps=20)

    if len(pl_hist_df['NetProfitMargin'][:i+1]) < 8:
        pm_forecast = SimpleExpSmoothing(pl_hist_df['NetProfitMargin'][:i+1].ffill()).fit().forecast(steps=20)
    else:
        pm_forecast = ExponentialSmoothing(pl_hist_df['NetProfitMargin'][:i+1].ffill(), trend='add', seasonal='add', seasonal_periods=4).fit().forecast(steps=20)

    # Create a forecast DataFrame for this specific historical point
    forecast_df = pd.DataFrame({
        'TotalRevenueForecast': revenue_forecast.values,
        'NetProfitMarginForecast': pm_forecast.values,
        'DilutedAverageSharesForecast': shares_forecast.values
    }, index=forecast_dates)

    # Calculate the forecasted Net Income
    forecast_df['NetIncomeForecast'] = forecast_df['TotalRevenueForecast'] * forecast_df['NetProfitMarginForecast']

    # Calculate the forecasted Net Income Per Share
    forecast_df['NetIncomePerShareForecast'] = forecast_df['NetIncomeForecast'] / forecast_df['DilutedAverageSharesForecast']

    # Calculate the discounted values for each period
    forecast_df['DiscountFactor'] = discount_factors
    forecast_df['DiscountedNetIncomePerShare'] = forecast_df['NetIncomePerShareForecast'] * forecast_df['DiscountFactor']

    # Terminal value calculation (using the last forecasted value)
    price_to_sales_ratio =.60
    terminal_value = (forecast_df['TotalRevenueForecast'].iloc[-1] * price_to_sales_ratio * 4) / forecast_df['DilutedAverageSharesForecast'].iloc[-1]
    terminal_value_discounted = terminal_value / ((1 + wacc) ** 5)

    # Sum the discounted cash flows and terminal value to calculate NPV
    npv_cashflows = forecast_df['DiscountedNetIncomePerShare'].sum()
    npv_terminal = terminal_value_discounted
    npv = npv_cashflows + npv_terminal

    # Append the result to historical NPVs
    historical_npvs.append((current_date, npv))

# Create a DataFrame for historical NPV values
historical_npv_df = pd.DataFrame(historical_npvs, columns=['Date', 'NPV'])

# Replace any remaining NaN values in NPV with 0
historical_npv_df['NPV'].fillna(0, inplace=True)





###################################


# Calculate Price-to-Book Ratio
balance_sheet_df['BookValuePerShare'] = (balance_sheet_df['TotalAssets'] - balance_sheet_df['TotalLiabilitiesNetMinorityInterest']) / pl_hist_df['DilutedAverageShares']
brk_b_df_quarterly = brk_b_df_quarterly.to_frame()
brk_b_df_quarterly['PriceToBook'] = brk_b_df_quarterly['Adj Close'] / balance_sheet_df['BookValuePerShare']


# Calculate z-scores for Price-to-Book ratios
brk_b_df_quarterly['PriceToBookZScore'] = (brk_b_df_quarterly['PriceToBook'] - brk_b_df_quarterly['PriceToBook'].mean()) / brk_b_df_quarterly['PriceToBook'].std()

# Trim outliers from the 'PriceToBook' column based on quantiles
lower_quantile = 0.05  # 5th percentile
upper_quantile = 0.95  # 95th percentile

# Apply clipping to remove extreme outliers
brk_b_df_quarterly['PriceToBookTrimmed'] = brk_b_df_quarterly['PriceToBook'].clip(
    lower=brk_b_df_quarterly['PriceToBook'].quantile(lower_quantile),
    upper=brk_b_df_quarterly['PriceToBook'].quantile(upper_quantile)
)

# Recalculate mean and standard deviation for the trimmed data
mean_trimmed = brk_b_df_quarterly['PriceToBookTrimmed'].mean()
std_trimmed = brk_b_df_quarterly['PriceToBookTrimmed'].std()

# Calculate Z-scores for the trimmed data
brk_b_df_quarterly['PriceToBookZScore'] = (
    (brk_b_df_quarterly['PriceToBookTrimmed'] - mean_trimmed) / std_trimmed
)

# Apply the color mapping function based on the updated Z-scores
def price_to_book_color(z_score):
    if z_score < -.5:
        return 'blue'  # Significantly undervalued
    elif -.5 <= z_score <= .5:
        return 'green'  # Fairly valued
    else:
        return 'red'  # Significantly overvalued

brk_b_df_quarterly['Color'] = brk_b_df_quarterly['PriceToBookZScore'].apply(price_to_book_color)
# Merge the 'Color' column from brk_b_df_quarterly into historical_npv_df
historical_npv_df = historical_npv_df.merge(
    brk_b_df_quarterly[['Color']], 
    left_on='Date', 
    right_index=True, 
    how='left'
)


# Plot Revenue and Net Income
fig1, ax1 = plt.subplots(figsize=(14, 8))

# Plot Revenue and Net Income on the primary y-axis
ax1.set_xlabel('Date')
ax1.set_ylabel('Revenue and Net Income', color='blue')
ax1.plot(pl_hist_df.index, pl_hist_df['TotalRevenue'], color='blue', label='Total Revenue', zorder=2)
ax1.plot(pl_hist_df.index, pl_hist_df['NetIncome'], color='green', label='Net Income', zorder=2)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.legend(loc='upper left')

fig1.tight_layout()
plt.show()

# Plot Adj Close Price and NPV on the same axis
fig2, ax2 = plt.subplots(figsize=(14, 8))

# Plot Adj Close Price (Existing Plot)
ax2.set_xlabel('Date')
ax2.set_ylabel('Adj Close Price and NPV', color='orange')
ax2.plot(brk_b_df_quarterly.index, brk_b_df_quarterly['Adj Close'], color='orange', label='Adj Close Price', zorder=2)

for i in range(len(historical_npv_df) - 1):
    if pd.notna(historical_npv_df['Color'].iloc[i]):  # Ensure there's a valid color to use
        ax2.plot(
            historical_npv_df['Date'].iloc[i:i + 2],
            historical_npv_df['NPV'].iloc[i:i + 2],
            color=historical_npv_df['Color'].iloc[i],  # Now correctly sourced from historical_npv_df
            linewidth=2,
            label='NPV' if i == 0 else ""
        )

ax2.tick_params(axis='y', labelcolor='orange')
ax2.legend(loc='upper left')
fig2.tight_layout()
plt.show()

print(wacc)
