import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from matplotlib.ticker import FuncFormatter

# API initialization
fred = Fred(api_key='24038268c5bd55ec155497b0079b5eb4')

# Retrieve data from FRED API
vix_data = pd.DataFrame(fred.get_series('VIXCLS', observation_start='2007-01-01', index_col=0))

# Add column name to the DataFrame
vix_data.columns = ['VIXCLS']

# Calculate the 90th percentile and median of historical VIX data
vix_90th_percentile = vix_data['VIXCLS'].quantile(0.9)
vix_10th_percentile = vix_data['VIXCLS'].quantile(0.1)
vix_median = vix_data['VIXCLS'].median()

# User-defined price threshold
user_defined_threshold = 13  

# Store Values in List
print("90th Percentile:", vix_90th_percentile)
print("10th Percentile:", vix_10th_percentile)
print("Median:", vix_median)

# Find dates when VIX crosses the 10th and 90th percentiles and the user-defined threshold
vix_cross_10th = vix_data[vix_data['VIXCLS'] <= vix_10th_percentile].index
vix_cross_90th = vix_data[vix_data['VIXCLS'] >= vix_90th_percentile].index
user_defined_cross = vix_data[vix_data['VIXCLS'] <= user_defined_threshold].index

# Initialize lists to store time intervals
time_intervals_10th_to_median = []
time_intervals_90th_to_median = []
time_intervals_user_defined_to_median = []

# Initialize dictionaries to store yearly counts
yearly_counts_10th = {}
yearly_counts_90th = {}
yearly_counts_user_defined = {}

# Calculate time intervals and yearly counts
for date in vix_cross_10th:
    future_dates = vix_data[date:].index
    for future_date in future_dates:
        if vix_data['VIXCLS'][future_date] >= vix_median:
            time_intervals_10th_to_median.append((future_date - date).days)
            year = future_date.year
            yearly_counts_10th[year] = yearly_counts_10th.get(year, 0) + 1
            break

for date in vix_cross_90th:
    future_dates = vix_data[date:].index
    for future_date in future_dates:
        if vix_data['VIXCLS'][future_date] <= vix_median:
            time_intervals_90th_to_median.append((future_date - date).days)
            year = future_date.year
            yearly_counts_90th[year] = yearly_counts_90th.get(year, 0) + 1
            break

for date in user_defined_cross:
    future_dates = vix_data[date:].index
    for future_date in future_dates:
        if vix_data['VIXCLS'][future_date] >= vix_median:
            time_intervals_user_defined_to_median.append((future_date - date).days)
            year = future_date.year
            yearly_counts_user_defined[year] = yearly_counts_user_defined.get(year, 0) + 1
            break

# Function to format y-axis ticks as percentages
def percentage_formatter(x, pos):
    return f"{x:.0%}"

# Calculate the histogram counts for all three thresholds
hist_10th, bins_10th = np.histogram(time_intervals_10th_to_median, bins=40)
hist_90th, bins_90th = np.histogram(time_intervals_90th_to_median, bins=40)
hist_user_defined, bins_user_defined = np.histogram(time_intervals_user_defined_to_median, bins=40)

# Calculate total counts
total_counts_10th = len(time_intervals_10th_to_median)
total_counts_90th = len(time_intervals_90th_to_median)
total_counts_user_defined = len(time_intervals_user_defined_to_median)

# Calculate percentages
percentage_10th = hist_10th / total_counts_10th
percentage_90th = hist_90th / total_counts_90th
percentage_user_defined = hist_user_defined / total_counts_user_defined

# Plot histograms with exact percentages
plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1)
plt.bar(bins_10th[:-1], percentage_10th, width=np.diff(bins_10th), edgecolor='k')
plt.title('Time for 10th Percentile to Median')
plt.xlabel('Time Interval (Days)')
plt.ylabel('Percentage of Total Occurrences')
plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

plt.subplot(1, 3, 2)
plt.bar(bins_90th[:-1], percentage_90th, width=np.diff(bins_90th), edgecolor='k')
plt.title('Time for 90th Percentile to Median')
plt.xlabel('Time Interval (Days)')
plt.ylabel('Percentage of Total Occurrences')
plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

plt.subplot(1, 3, 3)
plt.bar(bins_user_defined[:-1], percentage_user_defined, width=np.diff(bins_user_defined), edgecolor='k')
plt.title(f'Time for {user_defined_threshold} Threshold to Median')
plt.xlabel('Time Interval (Days)')
plt.ylabel('Percentage of Total Occurrences')
plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

plt.tight_layout()

# Plot yearly counts
years_10th = list(yearly_counts_10th.keys())
counts_10th = list(yearly_counts_10th.values())

years_90th = list(yearly_counts_90th.keys())
counts_90th = list(yearly_counts_90th.values())

years_user_defined = list(yearly_counts_user_defined.keys())
counts_user_defined = list(yearly_counts_user_defined.values())

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(years_10th, counts_10th, marker='o')
plt.title('Yearly Counts for 10th Percentile Crossings')
plt.xlabel('Year')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
plt.plot(years_90th, counts_90th, marker='o')
plt.title('Yearly Counts for 90th Percentile Crossings')
plt.xlabel('Year')
plt.ylabel('Count')

plt.subplot(1, 3, 3)
plt.plot(years_user_defined, counts_user_defined, marker='o')
plt.title(f'Yearly Counts for {user_defined_threshold} Threshold Crossings')
plt.xlabel('Year')
plt.ylabel('Count')

plt.tight_layout()
plt.show()
