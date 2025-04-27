import json
import numpy as np
import pandas as pd
import math
from typing import Any, Dict, List
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation
import string
from statistics import NormalDist

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 2000

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    DJEMBES = "DJEMBES"
    SPREAD = "SPREAD"
    SYNTHETIC = "SYNTHETIC"
    SYNTHETIC_B2 = "SYNTHETIC_B2"
    SPREAD_B2 = "SPREAD_B2"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"
    SPREAD_STOP = "SPREAD_STOP"


PARAMS = {
    Product.SPREAD: {
        "default_spread_mean": 58,
        "default_spread_std": 78.66710516685775,
        "spread_std_window": 160,
        "zscore_threshold": 2.711135,
        "target_position": 60,
        "relative": 0,
        },
    Product.SPREAD_B2: {
        "default_spread_mean": 30.33,
        "default_spread_std": 51.88543913150766,
        "spread_std_window": 110,
        "zscore_threshold": 2.604117,
        "target_position": 180,
        "relative": 0,
    },
    Product.MAGNIFICENT_MACARONS:{
        "make_probability": 0.566,
        "init_make_edge": 3.361185983827493,
        "min_edge": 1.5,
        "volume_avg_timestamp": 5,
        "volume_bar": 75,
        "dec_edge_discount": 0.8,
        "step_size":0.5
    },
    Product.SPREAD_STOP{
        "default_spread_mean": 30.33,
        "default_spread_std": 51.88543913150766,
        "spread_std_window": 110,
        "zscore_threshold": 2.604117,
        "target_position": 180,
        "relative": 0,
    },
}



#######################################################
#######################################################
###########                             ###############
###########          BIG ONE            ###############
###########                             ###############
#######################################################
#######################################################

class Trader:
    
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

    POSITION_LIMITS = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50,
        "SQUID_INK": 50,
        "CROISSANTS": 250,
        "JAMS": 350,
        "PICNIC_BASKET1": 60,
        "PICNIC_BASKET2": 100,
        "DJEMBES": 60,
        "VOLCANIC_ROCK": 400,
        "VOLCANIC_ROCK_VOUCHER_9500": 200,
        "VOLCANIC_ROCK_VOUCHER_9750": 200,
        "VOLCANIC_ROCK_VOUCHER_10000": 200,
        "VOLCANIC_ROCK_VOUCHER_10250": 200,
        "VOLCANIC_ROCK_VOUCHER_10500": 200,
        "MAGNIFICENT_MACARONS": 75,
    }

    acceptable_prices = {}

    kelp_price_hist = list()
    ink_price_hist = list()
    ink_vol_sum = 0
    ink_vol_ctr = 0
    ink_EMA = 0

    

    options_spread_hist = {}
    options_mid_price_hist = {}
    options_vol_spread_hist = {}


    # TRACKS SUNLIGHT HISTORY #
    sunlight = [0,0,0]

    # MID PRICE HISTORY FOR MACARONS #
    macaron_price_hist = list()

    # CURRENT CSI SIGNAL #
    csi_signal = 0

    def logger_print(self, product, width, pred, current_position, limit, cheapest):
        logger.print(f"{product} - Fair Bid: {pred-width}, Current position: {current_position}, Limit: {limit}, Cheapest Ask: {cheapest[0]}\n")
        logger.print(f"{product} - Fair Ask: {pred+width}, Current position: {current_position}, Limit: {limit}, Cheapest Bid: {cheapest[1]}\n")
    #############################################################################
    #############################################################################
    ################    PRICING HELPER FUNCTIONS    #############################
    #############################################################################
    #############################################################################
    def vwap(self, state):

        prices = {}

        for prod, order_depth in state.order_depths.items():

            # Only executes if there are both sell orders and buy orders;
            # otherwise, returns the previous state's mid point
            if len(order_depth.sell_orders) > 0 or len(order_depth.buy_orders) > 0:
    
                weighted_price_sum = 0
                volume_sum = 0
                for x in list(order_depth.sell_orders.items()) + list(order_depth.buy_orders.items()):
                    weighted_price_sum += x[0]*abs(x[1])
                    volume_sum += abs(x[1])

                prices[prod] = weighted_price_sum/volume_sum
                
            else:
                prices[prod] = self.acceptable_prices[prod]
                
        return prices

    ### BASKET 1 HELP FUNCTIONS ####

    def vwap_single(self, order_depth: OrderDepth) -> float:
        if len(order_depth.sell_orders) > 0 or len(order_depth.buy_orders) > 0:
    
                weighted_price_sum = 0
                volume_sum = 0
                for x in list(order_depth.sell_orders.items()) + list(order_depth.buy_orders.items()):
                    weighted_price_sum += x[0]*abs(x[1])
                    volume_sum += abs(x[1])

                return weighted_price_sum/volume_sum


    def synthetic_basket1_order_depth (self, order_depths: Dict[str, OrderDepth]):
        synthetic_order_depth = OrderDepth()

        CROISSANTS_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        CROISSANTS_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        JAMS_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        JAMS_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )
        DJEMBES_best_bid = (
            max(order_depths[Product.DJEMBES].buy_orders.keys())
            if order_depths[Product.DJEMBES].buy_orders
            else 0
        )
        DJEMBES_best_ask = (
            min(order_depths[Product.DJEMBES].sell_orders.keys())
            if order_depths[Product.DJEMBES].sell_orders
            else float("inf")
        )

        implied_bid = (
            CROISSANTS_best_bid * 6
            + JAMS_best_bid * 3
            + DJEMBES_best_bid * 1
        )
        implied_ask = (
            CROISSANTS_best_bid * 6
            + JAMS_best_bid * 3
            + DJEMBES_best_bid * 1
        )

        if implied_bid > 0:
            CROISSANTS_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[CROISSANTS_best_bid]
                // 6
                )
            JAMS_bid_volume = (
                order_depths[Product.JAMS].buy_orders[JAMS_best_bid]
                // 3
                )
            DJEMBES_bid_volume = (
                order_depths[Product.DJEMBES].buy_orders[DJEMBES_best_bid]
                // 1
                )
            implied_bid_volume = min(
                CROISSANTS_bid_volume, JAMS_bid_volume, DJEMBES_bid_volume
                )
            synthetic_order_depth.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            CROISSANTS_bid_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[CROISSANTS_best_ask]
                // 6
                )
            JAMS_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[JAMS_best_ask]
                // 3
                )
            DJEMBES_ask_volume = (
                -order_depths[Product.DJEMBES].sell_orders[DJEMBES_best_ask]
                // 2
                )
            implied_ask_volume = min(
                CROISSANTS_bid_volume, JAMS_ask_volume, DJEMBES_ask_volume
                )
            synthetic_order_depth.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_depth



    def convert_synthetic_basket1_orders (self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
      ):
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBES: [],
        }
        synthetic_basket1_order_depth = self.synthetic_basket1_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic_basket1_order_depth.buy_orders.keys())
            if synthetic_basket1_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket1_order_depth.sell_orders.keys())
            if synthetic_basket1_order_depth.sell_orders
            else float("inf")
        )

        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                CROISSANTS_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                JAMS_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
                DJEMBES_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                CROISSANTS_price = max(order_depths[Product.CROISSANTS].buy_orders.keys()
                )
                JAMS_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
                DJEMBES_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            CROISSANTS_order = Order(
                Product.CROISSANTS,
                CROISSANTS_price,
                quantity * 6,
            )
            JAMS_order = Order(
                Product.JAMS,
                JAMS_price,
                quantity * 3,
            )
            DJEMBES_order = Order(
                Product.DJEMBES, DJEMBES_price, quantity * 2
            )

            # Add the component orders to the respective lists
            component_orders[Product.CROISSANTS].append(CROISSANTS_order)
            component_orders[Product.JAMS].append(JAMS_order)
            component_orders[Product.DJEMBES].append(DJEMBES_order)

        return component_orders



    ### BASKET 2 HELP FUNCTIONS####

    def synthetic_basket2_order_depth (self, order_depths: Dict[str, OrderDepth]):
        synthetic_order2_depth = OrderDepth()

        CROISSANTS_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        CROISSANTS_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        JAMS_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        JAMS_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )

        implied_bid = (
            CROISSANTS_best_bid * 4
            + JAMS_best_bid * 2

        )
        implied_ask = (
            CROISSANTS_best_bid * 4
            + JAMS_best_bid * 2

        )

        if implied_bid > 0:
            CROISSANTS_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[CROISSANTS_best_bid]
                // 4
                )
            JAMS_bid_volume = (
                order_depths[Product.JAMS].buy_orders[JAMS_best_bid]
                // 2
                )
            implied_bid_volume = min(
                CROISSANTS_bid_volume, JAMS_bid_volume
                )
            synthetic_order2_depth.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            CROISSANTS_bid_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[CROISSANTS_best_ask]
                // 4
                )
            JAMS_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[JAMS_best_ask]
                // 2
                )
            implied_ask_volume = min(
                CROISSANTS_bid_volume, JAMS_ask_volume
                )
            synthetic_order2_depth.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order2_depth

    def convert_synthetic_basket2_orders (self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
      ):
          component_orders = {
              Product.CROISSANTS: [],
              Product.JAMS: [],
              Product.DJEMBES: [],
          }
          synthetic_basket2_order_depth = self.synthetic_basket2_order_depth(
              order_depths
          )
          best_bid = (
              max(synthetic_basket2_order_depth.buy_orders.keys())
              if synthetic_basket2_order_depth.buy_orders
              else 0
          )
          best_ask = (
              min(synthetic_basket2_order_depth.sell_orders.keys())
              if synthetic_basket2_order_depth.sell_orders
              else float("inf")
          )

          for order in synthetic_orders:
              # Extract the price and quantity from the synthetic basket order
              price = order.price
              quantity = order.quantity

              # Check if the synthetic basket order aligns with the best bid or ask
              if quantity > 0 and price >= best_ask:
                  # Buy order - trade components at their best ask prices
                  CROISSANTS_price = min(
                      order_depths[Product.CROISSANTS].sell_orders.keys()
                  )
                  JAMS_price = min(
                      order_depths[Product.JAMS].sell_orders.keys()
                  )

              elif quantity < 0 and price <= best_bid:
                  # Sell order - trade components at their best bid prices
                  CROISSANTS_price = max(order_depths[Product.CROISSANTS].buy_orders.keys()
                  )
                  JAMS_price = max(
                      order_depths[Product.JAMS].buy_orders.keys()
                  )

              else:
                  # The synthetic basket order does not align with the best bid or ask
                  continue

              # Create orders for each component
              CROISSANTS_order = Order(
                  Product.CROISSANTS,
                  CROISSANTS_price,
                  quantity * 4,
              )
              JAMS_order = Order(
                  Product.JAMS,
                  JAMS_price,
                  quantity * 2,
              )


              # Add the component orders to the respective lists
              component_orders[Product.CROISSANTS].append(CROISSANTS_order)
              component_orders[Product.JAMS].append(JAMS_order)


          return component_orders

    # SPREAD CREATOR FOR BOTH BOXES #

    def create_combined_spread_orders_baskets_1_and_2(
        self,
        #MAYBE ADD PRODUCT 
        order_depths: Dict[str, OrderDepth],
        position_b1: int,
        spread_data_b1: Dict[str, Any],
        position_b2: int,
        spread_data_b2: Dict[str, Any],
    ):
        # Skip if either product is missing
        if Product.PICNIC_BASKET1 not in order_depths or Product.PICNIC_BASKET2 not in order_depths:
            return None, None
        
        #Croissant - Jam Spread
        Croissant_vwap = self.vwap_single(order_depths[Product.CROISSANTS])
        Jam_vwap = self.vwap_single(order_depths[Product.JAMS])
        spread_stop = Croissant_vwap - Jam_vwap
        spread_data_stop["spread_history"].append(spread_stop)

        if (
            len(spread_data_stop["spread_history"]) < self.params[Product.SPREAD_STOP]["spread_std_window"]
            ):
                return None
        elif len(spread_data_stop["spread_history"]) > self.params[Product.SPREAD_STOP]["spread_std_window"]:
                spread_data_stop["spread_history"].pop(0)
        
        std_stop = np.std(spread_data_stop["spread_history"])
        z_stop = (spread_stop - self.params[Product.SPREAD_STOP]["default_spread_mean"]) / std_stop


        # Basket 1 spread
        b1_vwap = self.vwap_single(order_depths[Product.PICNIC_BASKET1])
        b1_synth_vwap = self.vwap_single(self.synthetic_basket1_order_depth(order_depths))
        spread_b1 = b1_vwap - b1_synth_vwap
        spread_data_b1["spread_history"].append(spread_b1)

        if (
            len(spread_data_b1["spread_history"]) < self.params[Product.SPREAD]["spread_std_window"]
            ):
                return None
        elif len(spread_data_b1["spread_history"]) > self.params[Product.SPREAD]["spread_std_window"]:
                spread_data_b1["spread_history"].pop(0)

        std_b1 = np.std(spread_data_b1["spread_history"])

        z_b1 = (spread_b1 - self.params[Product.SPREAD]["default_spread_mean"]) / std_b1

        # Basket 2 spread
        b2_vwap = self.vwap_single(order_depths[Product.PICNIC_BASKET2])
        b2_synth_vwap = self.vwap_single(self.synthetic_basket2_order_depth(order_depths))
        spread_b2 = b2_vwap - b2_synth_vwap
        spread_data_b2["spread_history"].append(spread_b2)

        if len(spread_data_b2["spread_history"]) > self.params[Product.SPREAD_B2]["spread_std_window"]:
            spread_data_b2["spread_history"].pop(0)
        if len(spread_data_b2["spread_history"]) < self.params[Product.SPREAD_B2]["spread_std_window"]:
            return None, None

        std_b2 = np.std(spread_data_b2["spread_history"])
        z_b2 = (spread_b2 - self.params[Product.SPREAD_B2]["default_spread_mean"]) / std_b2

        # Liquidate basket 2 if z_b2 is close to 0
        if -0.1 < z_b1 < 0.1 and position_b1 != 0:
            basket1_orders = self.execute_PICNIC_BASKET1_orders(
                0,
                position_b1,
                order_depths,
            )

        
        if -0.1 < z_b2 < 0.1 and position_b2 != 0:
            basket2_orders = self.execute_PICNIC_BASKET2_orders(
                0,
                position_b2,
                order_depths,
            )

        # Basket 1 decision
        basket1_orders = None
        if z_b1 >= self.params[Product.SPREAD]["zscore_threshold"] and z_stop <= self.params[Product.SPREAD]["zscore_threshold"]:
            if position_b1 != -self.params[Product.SPREAD]["target_position"]:
                basket1_orders = self.execute_PICNIC_BASKET1_orders(
                    -self.params[Product.SPREAD]["target_position"],
                    position_b1,
                    order_depths,
                )
        elif z_b1 <= -self.params[Product.SPREAD]["zscore_threshold"] and z_stop => -self.params[Product.SPREAD]["zscore_threshold"]:
            if position_b1 != self.params[Product.SPREAD]["target_position"]:
                basket1_orders = self.execute_PICNIC_BASKET1_orders(
                    self.params[Product.SPREAD]["target_position"],
                    position_b1,
                    order_depths,
                )

        # Basket 2 decision (only if same sign as basket 1 and threshold is triggered)
        basket2_orders = None
        if (spread_b2 * spread_b1 > 0):  # same sign
            if z_b2 >= self.params[Product.SPREAD_B2]["zscore_threshold"] and z_stop <= self.params[Product.SPREAD]["zscore_threshold"]:
                if position_b2 != -self.params[Product.SPREAD_B2]["target_position"]:
                    basket2_orders = self.execute_PICNIC_BASKET2_orders(
                        -self.params[Product.SPREAD_B2]["target_position"],
                        position_b2,
                        order_depths,
                    )
            elif z_b2 <= -self.params[Product.SPREAD_B2]["zscore_threshold"] and z_stop => -self.params[Product.SPREAD]["zscore_threshold"]:
                if position_b2 != self.params[Product.SPREAD_B2]["target_position"]:
                    basket2_orders = self.execute_PICNIC_BASKET2_orders(
                        self.params[Product.SPREAD_B2]["target_position"],
                        position_b2,
                        order_depths,
                    )

        spread_data_b1["prev_zscore"] = z_b1
        spread_data_b2["prev_zscore"] = z_b2
        return basket1_orders, basket2_orders

            ##### BOX 1 ORDERS #####

    def execute_PICNIC_BASKET1_orders(
            self,
            target_position: int,
            basket_position: int,
            order_depths: Dict[str, OrderDepth],
        ):

            if target_position == basket_position:
                return None

            target_quantity = abs(target_position - basket_position)
            basket_order_depth = order_depths[Product.PICNIC_BASKET1]
            synthetic_order_depth = self.synthetic_basket1_order_depth(order_depths)

            if target_position > basket_position:
                basket_ask_price = min(basket_order_depth.sell_orders.keys())
                basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

                synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
                synthetic_bid_volume = abs(
                    synthetic_order_depth.buy_orders[synthetic_bid_price]
                )

                orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
                execute_volume = min(orderbook_volume, target_quantity)

                basket_orders = [
                    Order(Product.PICNIC_BASKET1, basket_ask_price, execute_volume)
                ]
                synthetic_orders = [
                    Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)
                ]

                aggregate_orders = self.convert_synthetic_basket1_orders(
                    synthetic_orders, order_depths
                )
                aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
                return aggregate_orders

            else:
                basket_bid_price = max(basket_order_depth.buy_orders.keys())
                basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

                synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
                synthetic_ask_volume = abs(
                    synthetic_order_depth.sell_orders[synthetic_ask_price]
                )

                orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
                execute_volume = min(orderbook_volume, target_quantity)

                basket_orders = [
                    Order(Product.PICNIC_BASKET1, basket_bid_price, -execute_volume)
                ]
                synthetic_orders = [
                    Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)
                ]

                aggregate_orders = self.convert_synthetic_basket1_orders(
                    synthetic_orders, order_depths
                )
                aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
                return aggregate_orders


    ##### BOX 2 ORDERS #####

    def execute_PICNIC_BASKET2_orders(
            self,
            target_position: int,
            basket_position: int,
            order_depths: Dict[str, OrderDepth],
        ):

            if target_position == basket_position:
                return None

            target_quantity = abs(target_position - basket_position)
            basket_order_depth = order_depths[Product.PICNIC_BASKET2]
            synthetic_order_depth = self.synthetic_basket2_order_depth(order_depths)

            if target_position > basket_position:
                basket_ask_price = min(basket_order_depth.sell_orders.keys())
                basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

                synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
                synthetic_bid_volume = abs(
                    synthetic_order_depth.buy_orders[synthetic_bid_price]
                )

                orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
                execute_volume = min(orderbook_volume, target_quantity)

                basket_orders = [
                    Order(Product.PICNIC_BASKET2, basket_ask_price, execute_volume)
                ]
                synthetic_orders = [
                    Order(Product.SYNTHETIC_B2, synthetic_bid_price, -execute_volume)
                ]

                aggregate_orders = self.convert_synthetic_basket2_orders(
                    synthetic_orders, order_depths
                )
                aggregate_orders[Product.PICNIC_BASKET2] = basket_orders
                return aggregate_orders

            else:
                basket_bid_price = max(basket_order_depth.buy_orders.keys())
                basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

                synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
                synthetic_ask_volume = abs(
                    synthetic_order_depth.sell_orders[synthetic_ask_price]
                )

                orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
                execute_volume = min(orderbook_volume, target_quantity)

                basket_orders = [
                    Order(Product.PICNIC_BASKET2, basket_bid_price, -execute_volume)
                ]
                synthetic_orders = [
                    Order(Product.SYNTHETIC_B2, synthetic_ask_price, execute_volume)
                ]

                aggregate_orders = self.convert_synthetic_basket2_orders(
                    synthetic_orders, order_depths
                )
                aggregate_orders[Product.PICNIC_BASKET2] = basket_orders
                return aggregate_orders


    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input,
        # and outputs a list of orders to be sent.
        # logger.print("traderData: " + state.traderData)
        # logger.print("Observations: " + str(state.observations))
        result = {}
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)


        self.acceptable_prices = self.vwap(state)

        ### MAKE ORDERS ###

        # result["RAINFOREST_RESIN"] = self.execute_resin_orders(state.order_depths["RAINFOREST_RESIN"], state)
        # result["KELP"] = self.execute_kelp_orders(state.order_depths["KELP"], state)
        # result["SQUID_INK"] = self.execute_ink_orders(state.order_depths["SQUID_INK"], state)

        if Product.SPREAD not in traderObject:
                traderObject[Product.SPREAD] = {
                    "spread_history": [],
                    "prev_zscore": 0,
                }
        if Product.SPREAD_B2 not in traderObject:
                traderObject[Product.SPREAD_B2] = {
                    "spread_history": [],
                    "prev_zscore": 0,
                }

        basket_position_1 = (
            state.position[Product.PICNIC_BASKET1]
            if Product.PICNIC_BASKET1 in state.position
            else 0
            )
        basket_position_2 = (
            state.position[Product.PICNIC_BASKET2]
            if Product.PICNIC_BASKET2 in state.position
            else 0
            )

        combined_result = self.create_combined_spread_orders_baskets_1_and_2(
            state.order_depths,
            basket_position_1,
            traderObject[Product.SPREAD],
            basket_position_2,
            traderObject[Product.SPREAD_B2]
        )

        if combined_result is not None:
            basket1_orders, basket2_orders = combined_result
        else:
            basket1_orders, basket2_orders = None, None


        if basket1_orders:
            result[Product.CROISSANTS] = basket1_orders.get(Product.CROISSANTS, [])
            result[Product.JAMS] = basket1_orders.get(Product.JAMS, [])
            result[Product.DJEMBES] =  basket1_orders.get(Product.DJEMBES, [])
            result[Product.PICNIC_BASKET1] = basket1_orders.get(Product.PICNIC_BASKET1, [])

        if basket2_orders:
            # Use += if CROISSANTS/JAMS were already in basket1_orders
            result[Product.CROISSANTS] = result.get(Product.CROISSANTS, []) + basket2_orders.get(Product.CROISSANTS, [])
            result[Product.JAMS] = result.get(Product.JAMS, []) + basket2_orders.get(Product.JAMS, [])
            result[Product.PICNIC_BASKET2] = basket2_orders.get(Product.PICNIC_BASKET2, [])

        # # for x,y in self.execute_options(state, state.order_depths).items():
        # #     # logger.print(x, y)
        # #     result[x] = [y]

        
        if Product.MAGNIFICENT_MACARONS in self.params and Product.MAGNIFICENT_MACARONS in state.order_depths:
            if "MAGNIFICENT_MACARONS" not in traderObject:
                traderObject["MAGNIFICENT_MACARONS"] = {"curr_edge": self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"], "volume_history": [], "optimized": False}
            macarons_position = (
                state.position[Product.MAGNIFICENT_MACARONS]
                if Product.MAGNIFICENT_MACARONS in state.position
                else 0
            )
            logger.print(f"MACARONS POSITION: {macarons_position}")

            conversions = self.macarons_arb_clear(
                macarons_position
            )

            adap_edge = self.macarons_adap_edge(
                state.timestamp,
                traderObject["MAGNIFICENT_MACARONS"]["curr_edge"],
                macarons_position,
                traderObject,
            )

            macarons_position = 0

            result[Product.MAGNIFICENT_MACARONS] = self.execute_CSI_trading(state, state.order_depths.get("MAGNIFICENT_MACARONS", OrderDepth()), state.observations.conversionObservations["MAGNIFICENT_MACARONS"])

            macarons_take_orders, buy_order_volume, sell_order_volume = self.macarons_arb_take(
                state.order_depths[Product.MAGNIFICENT_MACARONS],
                state.observations.conversionObservations[Product.MAGNIFICENT_MACARONS],
                adap_edge,
                macarons_position,
            )

            macarons_make_orders, _, _ = self.macarons_arb_make(
                state.order_depths[Product.MAGNIFICENT_MACARONS],
                state.observations.conversionObservations[Product.MAGNIFICENT_MACARONS],
                macarons_position,
                adap_edge,
                buy_order_volume,
                sell_order_volume
            )

            result[Product.MAGNIFICENT_MACARONS] = (
                result[Product.MAGNIFICENT_MACARONS] + macarons_take_orders + macarons_make_orders
            )


        traderData = jsonpickle.encode(traderObject)

        conversions = max(-10, min(10, conversions))
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
