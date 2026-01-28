import pandas as pd

# Load dataset
df = pd.read_csv("diabetes.csv")

# Show first 5 rows
print(df.head())

# Dataset info
print("\nINFO:")
print(df.info())

# Basic statistics
print("\nSTATS:")
print(df.describe())
