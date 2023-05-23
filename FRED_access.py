import pandas as pd
from fredapi import Fred

# API initialization
fred = Fred(api_key='xxxxx')

# Retrieve data from FRED API
jpy_usd = pd.DataFrame(fred.get_series('EXJPUS', observation_start='1971-01-01', index_col=0))
jpy_usd = jpy_usd.rename(columns={0: 'EXJPUS'})

oil_rate = pd.DataFrame(fred.get_series('PCEDG', observation_start='1971-01-01', index_col=0))
oil_rate = oil_rate.rename(columns={0: 'oil'})

usdrate=pd.DataFrame(fred.get_series('FEDFUNDS',observation_start = '2010-01-01',index_col=0))
usdrate = usdrate.rename(columns={0: 'rate'})


# Merge the data frames based on the index (dates)
merged_df = pd.merge(jpy_usd, oil_rate, left_index=True, right_index=True)
merged_df = pd.merge(merged_df, usdrate, left_index=True, right_index=True)

# Print the head of the df
print(merged_df.head)
