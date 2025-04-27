import pandas as pd
import matplotlib.pyplot as plt

# Re-read the CSVs with correct delimiter `;`
prices_df = pd.read_csv("prices_round_4_day_1.csv", delimiter=";")
trades_df = pd.read_csv("trades_round_4_day_1.csv", delimiter=";")
observations_df = pd.read_csv("observations_round_4_day_1.csv", delimiter=",")




macaron_prices = prices_df[prices_df['product'] == "MAGNIFICENT_MACARONS"]
macaron_trades = trades_df[trades_df['symbol'] == "MAGNIFICENT_MACARONS"]
merged_df = pd.merge(macaron_prices, observations_df, on="timestamp", how="inner")
merged_df2 = pd.merge(macaron_prices, macaron_trades, on="timestamp", how="inner")

#spread one
merged_df["implied_ask"] = merged_df["askPrice"] + merged_df["importTariff"] + merged_df["transportFees"]
spread = merged_df["bid_price_1"] - merged_df["implied_ask"] + 1.5
spread_mean = spread.mean()
print(spread_mean)

#spread two 
spread2 = merged_df2["price"] - merged_df2["bid_price_1"]
spread2_mean = spread2.mean()
print(spread2_mean)

### Extra ###
mid_price = macaron_prices["mid_price"]
bid_price_1 = macaron_prices["bid_price_1"]
bid_price_2 = macaron_prices["bid_price_2"]
ask_price_1 = macaron_prices["ask_price_1"]
ask_price_2 = macaron_prices["ask_price_2"]

market_price = macaron_trades["price"]

# 
bid_mean = bid_price_1.mean()
ask_mean = ask_price_1.mean()

#Foreign Data
foreign_bid_mean = observations_df["bidPrice"].mean()
foreign_ask_mean = observations_df["askPrice"].mean()
implied_ask =  observations_df["askPrice"] - observations_df["importTariff"] + observations_df["transportFees"]


#Trading Data
market_price_mean = market_price.mean()

#calculating the spread 

# plt.figure(figsize=(12, 4))
# plt.plot(macaron_prices["timestamp"], bid_price_1, label="Best Bid Price")
# plt.title("MAGNIFICENT_MACARONS - Bid Price Over Time")
# plt.xlabel("Timestamp")
# plt.ylabel("Price")
# plt.grid(True)
# plt.show()

#pritn
# print(bid_mean)
# print(ask_mean)
# print(foreign_bid_mean)
# print(foreign_ask_mean)
# print(market_price_mean)

# Plot bid, ask, and mid prices over time


# plt.figure(figsize=(12, 4))
# plt.plot(macaron_prices["timestamp"], bid_price_1, label="Best Bid Price")
# plt.title("MAGNIFICENT_MACARONS - Bid Price Over Time")
# plt.xlabel("Timestamp")
# plt.ylabel("Price")
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(12, 4))
# plt.plot(macaron_prices["timestamp"], ask_price_1, label="Best Ask Price", color='orange')
# plt.title("MAGNIFICENT_MACARONS - Ask Price Over Time")
# plt.xlabel("Timestamp")
# plt.ylabel("Price")
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(12, 4))
# plt.plot(macaron_prices["timestamp"], mid_price, label="Mid Price", color='black', linestyle='--')
# plt.title("MAGNIFICENT_MACARONS - Mid Price Over Time")
# plt.xlabel("Timestamp")
# plt.ylabel("Price")
# plt.grid(True)
# plt.show()

