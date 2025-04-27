import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# === Load Data ===
all_prices = pd.read_csv("prices_round_2_day_0.csv", sep=";")


# === Filter Relevant Products ===
relevant_products = ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1", "PICNIC_BASKET2"]
filtered_df = all_prices[all_prices["product"].isin(relevant_products)]

# === Average Mid Prices (in case of duplicates) ===
filtered_avg = (
    filtered_df.groupby(["timestamp", "product"])["mid_price"]
    .mean()
    .reset_index()
)

# === Pivot Table ===
pivot_df = filtered_avg.pivot(index="timestamp", columns="product", values="mid_price").sort_index()

# === Compute Synthetic Basket Prices ===
pivot_df["BASKET1_SYNTH"] = 6 * pivot_df["CROISSANTS"] + 3 * pivot_df["JAMS"] + 1 * pivot_df["DJEMBES"]
pivot_df["BASKET2_SYNTH"] = 4 * pivot_df["CROISSANTS"] + 2 * pivot_df["JAMS"]

# === Calculate Spread ===
pivot_df["SPREAD1"] = pivot_df["PICNIC_BASKET1"] - pivot_df["BASKET1_SYNTH"]
pivot_df["SPREAD2"] = pivot_df["PICNIC_BASKET2"] - pivot_df["BASKET2_SYNTH"]
pivot_df["SPREAD3"] = pivot_df["PICNIC_BASKET1"] - pivot_df["DJEMBES"] - (1.5) * pivot_df["PICNIC_BASKET2"]

# === Z-Score Calculation ===
window = 50  # rolling window size
for i in [1, 3]:
    spread = f"SPREAD{i}"
    pivot_df[f"{spread}_MEAN"] = pivot_df[spread].rolling(window=window).mean()
    pivot_df[f"{spread}_STD"] = pivot_df[spread].rolling(window=window).std()
    pivot_df[f"{spread}_ZSCORE"] = (
        (pivot_df[spread] - pivot_df[f"{spread}_MEAN"]) / pivot_df[f"{spread}_STD"]
    )



default_spread_mean = pivot_df["SPREAD1"].mean()
default_spread_std = pivot_df["SPREAD1"].std()
default_spread2_mean = pivot_df["SPREAD2"].mean()
default_spread2_std = pivot_df["SPREAD2"].std()
default_spreadinter_mean = pivot_df["SPREAD3"].mean()
default_spreadinter_sd = pivot_df["SPREAD3"].std()


print(default_spread_mean)
print(default_spread_std)
print(default_spread2_mean)
print(default_spread2_std)
print(default_spreadinter_mean)
print(default_spreadinter_sd)
print(pivot_df["SPREAD1_ZSCORE"].quantile([0.90, 0.95, 0.975, 0.99]))
print(pivot_df["SPREAD3_ZSCORE"].quantile([0.90, 0.95, 0.975, 0.99]))
