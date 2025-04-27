import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# === Load Data ===
all_prices = pd.read_csv("prices_round_3_day_2.csv", sep=";")


# === Filter Relevant Products ===
relevant_products = ["SQUID_INK", "CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1", "PICNIC_BASKET2"]
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
pivot_df["SPREAD4"] = pivot_df["SQUID_INK"]
pivot_df["SPREAD5"] = pivot_df["DJEMBES"]
pivot_df["SPREAD6"] = pivot_df["JAMS"]
pivot_df["SPREAD7"] = pivot_df["CROISSANTS"]


# === Z-Score Calculation ===
window = 50  # rolling window size
for i in [1, 2, 3, 4]:
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
default_spreadInk_mean = pivot_df["SPREAD4"].mean()
default_spreadInk_sd = pivot_df["SPREAD4"].std()
jams_mean = pivot_df["SPREAD6"].std()
djembes_mean = pivot_df["SPREAD5"].std()
croissants_mean =pivot_df["SPREAD7"].std()

print(jams_mean)
print(djembes_mean)
print(croissants_mean)

print(pivot_df["SPREAD1_ZSCORE"].quantile([0.90, 0.95, 0.975, 0.99]))
print(pivot_df["SPREAD2_ZSCORE"].quantile([0.90, 0.95, 0.975, 0.99]))
print(pivot_df["SPREAD3_ZSCORE"].quantile([0.90, 0.95, 0.975, 0.99]))
print(pivot_df["SPREAD4_ZSCORE"].quantile([0.90, 0.95, 0.975, 0.99]))

#plots
fig1, ax1 = plt.subplots()
pivot_df[["SPREAD1", "SPREAD1_MEAN"]].plot(ax=ax1)
ax1.set_title("Spread1: Basket1 - Synthetic")
ax1.axhline(0, color="gray", linestyle="--")

fig2, ax2 = plt.subplots()
pivot_df["SPREAD1_ZSCORE"].plot(ax=ax2)
ax2.set_title("Z-score of Spread1")
ax2.axhline(1, color="red", linestyle="--")
ax2.axhline(-1, color="green", linestyle="--")

fig3, ax3 = plt.subplots()
pivot_df[["SPREAD2", "SPREAD2_MEAN"]].plot(ax=ax3)
ax3.set_title("Spread2: Basket2 - Synthetic")
ax3.axhline(0, color="gray", linestyle="--")

fig4, ax4 = plt.subplots()
pivot_df["SPREAD2_ZSCORE"].plot(ax=ax4)
ax4.set_title("Z-score of Spread2")
ax4.axhline(1, color="red", linestyle="--")
ax4.axhline(-1, color="green", linestyle="--")

plt.tight_layout()
plt.show()