import pandas as pd

# Load dataset
retail_data = pd.read_csv("~/Downloads/retail_price.csv")

# Define test prices for each category
category_test_prices = {
    'garden_tools': 100,
    'health_beauty': 1,
    'watches_gifts': 1,
    'computers_accessories': 1,
    'bed_bath_table': 1,
    'cool_stuff': 1,
    'furniture_decor': 1,
    'perfumery': 1,
    'consoles_games': 1
}

# Group by 'product_category_name' and perform calculations for each category
category_stats = []

for category, group in retail_data.groupby('product_category_name'):
    # Calculations for means
    avg_price = group["total_price"].mean()
    avg_demanded = group["qty"].mean()

    # Calculations for covariance
    group["price_diff"] = group["total_price"] - avg_price
    group["qty_diff"] = group["qty"] - avg_demanded
    covariance = (group["price_diff"] * group["qty_diff"]).sum() / (len(group) - 1)
    var_price = (group["price_diff"] ** 2).sum() / (len(group) - 1)

    # Calculate slope (b)
    slope_b = covariance / var_price

    # Calculate intercept (a)
    intercept_a = avg_demanded - (slope_b * avg_price)

    # Calculate PED
    ped = slope_b * (avg_price / avg_demanded)

    # Calculate demand at the test price for this category
    price_test = category_test_prices.get(category, avg_price)  # Default to avg_price if not specified
    demand_test = slope_b * price_test + intercept_a

    # Store the results
    category_stats.append({
        'category': category,
        'avg_price': avg_price,
        'avg_demanded': avg_demanded,
        'slope_b': slope_b,
        'intercept_a': intercept_a,
        'PED': ped,
        'price_test': price_test,
        'demand_test': demand_test
    })

# Convert to DataFrame for better readability
category_stats_df = pd.DataFrame(category_stats)

# Display the calculated statistics for each category
print(category_stats_df)

