import pandas as pd
from pandasql import sqldf

# Load data set
trans = pd.read_csv("~/Downloads/drive-download-20231128T140820Z-001 2/transactions.csv")
prod = pd.read_csv("~/Downloads/drive-download-20231128T140820Z-001 2/products.csv")
store = pd.read_csv("~/Downloads/drive-download-20231128T140820Z-001 2/stores.csv")

# Convert the "trans.price" column to decimal with two decimal places
trans['Price'] = pd.to_numeric(trans['Price'].str.replace('[\$,]', '', regex=True), errors='coerce')

pysqldf = lambda q: sqldf(q, globals())
query = """
    SELECT *
    FROM trans
    LEFT JOIN store ON store.StoreID = trans.StoreID
    LEFT JOIN prod ON prod.ProductID = trans.ProductID
"""

result = pysqldf(query)

# Save the result to a new CSV file
result.to_csv("~/Downloads/drive-download-20231128T140820Z-001 2/combined_data.csv", index=False)

print("Result saved to 'combined_data.csv'")
