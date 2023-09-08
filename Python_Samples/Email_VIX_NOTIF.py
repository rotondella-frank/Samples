import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from fredapi import Fred

# API initialization
fred = Fred(api_key='24038268c5bd55ec155497b0079b5eb4')

# Retrieve data from FRED API
vix_data = pd.DataFrame(fred.get_series('VIXCLS', observation_start='2020-01-01', index_col=0))

# Dropping rows with null values
vix_data = vix_data.dropna()

# Calculate the 90th and 10th percentiles of the VIX data
vix_values = vix_data.values  # Extract VIX values from DataFrame
threshold_above = int(np.percentile(vix_values, 90))  # 90th percentile
threshold_below = int(np.percentile(vix_values, 10))  # 10th percentile

# Calculate the prior day's date based on the day of the week
today = datetime.now()
if today.weekday() == 0:  # Monday
    prior_day = today - timedelta(days=3)  # Subtract 3 days to get to the prior Friday
else:
    prior_day = today - timedelta(days=1)  # Subtract 1 day to get to the prior day

prior_day_str = prior_day.strftime("%Y-%m-%d")

# Get the VIX value for the prior day
current_vix = vix_data.last_valid_index(vix_values)

# Check conditions and send email if necessary
if current_vix > threshold_above or current_vix < threshold_below:
    msg = MIMEMultipart()
    msg["From"] = "Rotondella.frank@gmail.com"
    msg["To"] = "Rotondella.frank@example.com"
    msg["Subject"] = "VIX Threshold Alert"

    body = f"Current VIX on {prior_day_str}: {current_vix}"
    msg.attach(MIMEText(body, "plain"))

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login("Rotondella.frank@gmail.com", "Rotondella.frank_PASSWORD")
    server.sendmail("Rotondella.frank@gmail.com", "Rotondella.frank@example.com", msg.as_string())
    server.quit()
