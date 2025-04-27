import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




trades1_df = pd.read_csv("trades_round_5_day_2.csv", delimiter=";")
trades2_df = pd.read_csv("trades_round_5_day_3.csv", delimiter=";")
trades3_df = pd.read_csv("trades_round_5_day_4.csv", delimiter=";")

trades_df = pd.concat([trades1_df, trades2_df, trades3_df], ignore_index=True)

# Determine trade direction from the perspective of each trader
def expand_trades(row):
    return pd.DataFrame([
        {'person': row['buyer'], 'symbol': row['symbol'], 'price': row['price'], 'quantity': row['quantity'], 'side': 'buy'},
        {'person': row['seller'], 'symbol': row['symbol'], 'price': row['price'], 'quantity': row['quantity'], 'side': 'sell'}
    ])

# Expand trades
expanded_trades = pd.concat([expand_trades(row) for _, row in trades_df.iterrows()], ignore_index=True)


for product in trades_df['symbol'].unique():
    # Filter and build buyer/seller data
    product_trades = trades_df[trades_df['symbol'] == product].copy()

    buyers = product_trades[['timestamp', 'buyer', 'symbol', 'quantity']].copy()
    buyers.columns = ['timestamp', 'person', 'product', 'quantity']
    buyers['side'] = 'buy'

    sellers = product_trades[['timestamp', 'seller', 'symbol', 'quantity']].copy()
    sellers.columns = ['timestamp', 'person', 'product', 'quantity']
    sellers['side'] = 'sell'

    expanded_trades = pd.concat([buyers, sellers], ignore_index=True)

    # Assign signed quantities
    expanded_trades['signed_quantity'] = expanded_trades.apply(
        lambda row: row['quantity'] if row['side'] == 'buy' else -row['quantity'], axis=1
    )

    # Sort by timestamp and compute cumulative depth
    expanded_trades.sort_values(by='timestamp', inplace=True)
    cumulative_depth = expanded_trades.groupby(['person', 'timestamp'])['signed_quantity']\
        .sum().groupby(level=0).cumsum().reset_index()

    if cumulative_depth.empty:
        continue  # skip plotting empty products

    pivot_df = cumulative_depth.pivot(index='timestamp', columns='person', values='signed_quantity').fillna(0)

    # Plot
    pivot_df.plot(figsize=(12, 6), title=f"Cumulative Order Depth by Person - {product}")
    plt.xlabel("Timestamp")
    plt.ylabel("Cumulative Net Quantity")
    plt.grid(True)
    plt.legend(title="Trader", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()