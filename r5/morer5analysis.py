import pandas as pd
import matplotlib.pyplot as plt

# Load the order book data
prices_df = pd.read_csv("prices_round_5_day_4.csv", delimiter=";")

# Clean column names
prices_df.columns = [col.strip() for col in prices_df.columns]

# Find all products
products = prices_df['product'].unique()

# Initialize a dictionary to store order depth over time
depth_data = {}
depth_change_data = {}

# Calculate order depth by summing bid and ask volumes
for product in products:
    product_df = prices_df[prices_df['product'] == product]
    product_df['order_depth'] = (
        product_df[['bid_volume_1', 'bid_volume_2', 'bid_volume_3']].sum(axis=1).abs() +
        product_df[['ask_volume_1', 'ask_volume_2', 'ask_volume_3']].sum(axis=1).abs()
    )
    product_df['order_depth_change'] = product_df['order_depth'].diff()
    depth_change_data[product] = product_df[['timestamp', 'order_depth_change']]

# Plot the order depth over time for each product
for product, data in depth_change_data.items():
    plt.figure()
    plt.plot(data['timestamp'], data['order_depth_change'])
    plt.title(f"Change in Order Depth Over Time - {product}")
    plt.xlabel("Timestamp")
    plt.ylabel("Change in Order Depth")
    plt.grid(True)
    plt.tight_layout()

plt.show()

