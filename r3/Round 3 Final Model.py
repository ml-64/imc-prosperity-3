import json
import numpy as np
import pandas as pd
import math
from typing import Any, Dict, List
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import string
from statistics import NormalDist

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

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
        "VOLCANIC_ROCK_VOUCHER_10500": 200
    }

    acceptable_prices = {}

    kelp_price_hist = list()
    ink_price_hist = list()
    ink_vol_sum = 0
    ink_vol_ctr = 0
    ink_EMA = 0

    def logger_print(self, product, width, pred, current_position, limit, cheapest):
        logger.print(f"{product} - Fair Bid: {pred-width}, Current position: {current_position}, Limit: {limit}, Cheapest Ask: {cheapest[0]}\n")
        logger.print(f"{product} - Fair Ask: {pred+width}, Current position: {current_position}, Limit: {limit}, Cheapest Bid: {cheapest[1]}\n")

    options_spread_hist = {}
    options_mid_price_hist = {}

    default_options_vol = {
        "VOLCANIC_ROCK_VOUCHER_9500": 9.250213706500483,
        "VOLCANIC_ROCK_VOUCHER_9750": 16.113672371581206,
        "VOLCANIC_ROCK_VOUCHER_10000": 22.150063341406153,
        "VOLCANIC_ROCK_VOUCHER_10250": 13.422148772131159,
        "VOLCANIC_ROCK_VOUCHER_10500": 14.443010012913295
    }


    #############################################################################
    #############################################################################
    ################    PRICING HELPER FUNCTIONS    #############################
    #############################################################################
    #############################################################################

    # DYNAMIC MARKET TAKING QUANTITY FOR RAINFOREST RESIN
    def dyn_limit_resin(self, price_diff):
        if price_diff >= 2: return 50
        else: return math.ceil(30 + 10 * price_diff)
        # return math.ceil(25+(25* (1-(0.5 ** float(price_diff)))))

    def dyn_limit_kelp(self, price_diff):
        if price_diff >= 5: return 50
        else: return math.ceil(10 * price_diff)

    def dyn_limit_ink(self, price_diff):
        return math.ceil(25+(25* (1-(0.5 ** (0.02 * float(price_diff))))))

    def pair_limit(self, price_diff):
        if price_diff > 0:
            return math.ceil(50*(1-0.5 ** (0.05 * float(price_diff))))
        if price_diff < 0:
            return -math.ceil(50*(1-0.5 ** (0.05 * float(-price_diff))))
        return 0

    # VOLUME WEIGHTED AVERAGE PRICING
    ### The Stanford 2024 team noticed that the true "mid price" tends to
    ### be determined by an automated market maker who sets a high volume
    ### on all their orders. This helper function takes the order depth and
    ### takes the average value, weighted in favor of orders with higher volume

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

        # Basket 1 decision
        basket1_orders = None
        if z_b1 >= self.params[Product.SPREAD]["zscore_threshold"]:
            if position_b1 != -self.params[Product.SPREAD]["target_position"]:
                basket1_orders = self.execute_PICNIC_BASKET1_orders(
                    -self.params[Product.SPREAD]["target_position"],
                    position_b1,
                    order_depths,
                )
        elif z_b1 <= -self.params[Product.SPREAD]["zscore_threshold"]:
            if position_b1 != self.params[Product.SPREAD]["target_position"]:
                basket1_orders = self.execute_PICNIC_BASKET1_orders(
                    self.params[Product.SPREAD]["target_position"],
                    position_b1,
                    order_depths,
                )

        # Basket 2 decision (only if same sign as basket 1 and threshold is triggered)
        basket2_orders = None
        if (spread_b2 * spread_b1 > 0):  # same sign
            if z_b2 >= self.params[Product.SPREAD_B2]["zscore_threshold"]:
                if position_b2 != -self.params[Product.SPREAD_B2]["target_position"]:
                    basket2_orders = self.execute_PICNIC_BASKET2_orders(
                        -self.params[Product.SPREAD_B2]["target_position"],
                        position_b2,
                        order_depths,
                    )
            elif z_b2 <= -self.params[Product.SPREAD_B2]["zscore_threshold"]:
                if position_b2 != self.params[Product.SPREAD_B2]["target_position"]:
                    basket2_orders = self.execute_PICNIC_BASKET2_orders(
                        self.params[Product.SPREAD_B2]["target_position"],
                        position_b2,
                        order_depths,
                    )

        spread_data_b1["prev_zscore"] = z_b1
        spread_data_b2["prev_zscore"] = z_b2
        return basket1_orders, basket2_orders

    ### MID PRICE ###
    def mid_price(self, order_depth):
        sells = len(order_depth.sell_orders.keys())
        buys = len(order_depth.buy_orders.keys())

        if sells > 0 and buys > 0:
            return (min(order_depth.sell_orders.keys()) + max(order_depth.buy_orders.keys()))/2

        elif sells > 0:
            return min(order_depth.sell_orders.keys())

        else:
            return max(order_depth.buy_orders.keys())

    ### BLACK SCHOLES ###
    def black_scholes_hist(self, S, K, timestamp):
        T = 5 - (timestamp / 1000000)  # Convert days to years
        sigma = 0.0268
        d1 = (np.log(S / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * NormalDist().cdf(d1) - K * NormalDist().cdf(d2)
        # print(norm.cdf(d2))
        return call_price

    def delta(self, S, K, timestamp):
        T = 5 - (timestamp / 1000000)
        sigma = 0.0268
        d1 = (np.log(S / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return NormalDist().cdf(d1)

        ### TRADING STRATEGY: TRADE ON MEAN REVERSION BETWEEN IMPLIED PRICE AND MARKET PRICE ###
    def options_signal1(self, state, order_depths):
        
        # FIND IMPLICIT PRICE #
        S = self.mid_price(order_depths["VOLCANIC_ROCK"])
        K = np.array([9500, 9750, 10000, 10250, 10500])
        timestamp = state.timestamp

        implicit_prices = [self.black_scholes_hist(S, x, timestamp) for x in K]

        # FIND MID PRICES #
        mid_prices = np.array([self.mid_price(order_depths["VOLCANIC_ROCK_VOUCHER_9500"]),
                               self.mid_price(order_depths["VOLCANIC_ROCK_VOUCHER_9750"]),
                               self.mid_price(order_depths["VOLCANIC_ROCK_VOUCHER_10000"]),
                               self.mid_price(order_depths["VOLCANIC_ROCK_VOUCHER_10250"]),
                               self.mid_price(order_depths["VOLCANIC_ROCK_VOUCHER_10500"])])

        # PUSH PRICES INTO HISTORY #
        def push(dict_type, key, spread):
            if dict_type.get(key):
                temp = dict_type.get(key)
                temp.append(spread)
                if len(temp) > 100:
                    temp.pop(0)
                dict_type[key] = temp
            else:
                dict_type[key] = [spread]
                
        for x in range(5):
            push(self.options_spread_hist, f"VOLCANIC_ROCK_VOUCHER_{str(9500 + x * 250)}", implicit_prices[x] - mid_prices[x])
            push(self.options_mid_price_hist, f"VOLCANIC_ROCK_VOUCHER_{str(9500 + x * 250)}", mid_prices[x])
            

        # TRADE THE OPTIONS #
        res = {}
        
        for x,y in self.options_spread_hist.items():

            prod = x
            
            # CALCULATE Z-SCORE ON A ROLLING BASIS #
            rolling_mean = np.mean(np.array(y))
            rolling_std = np.std(np.array(y))

            if rolling_std == 0:
                rolling_std = self.default_options_vol[x]

            z_score = (y[-1] - rolling_mean)/rolling_std

            # TRADE IF Z SCORE IS ABOVE 2
            if abs(z_score) > 2:

                # IMPLIED - MID PRICE is positive i.e. mid price is undervalued, so we buy
                if z_score > 2:
                    res[prod] = 1
                        
                else:
                    res[prod] = -1
            else:
                res[prod] = 0

            # logger.print(f"Product: {prod}\n    Mean: {rolling_mean}, SD: {rolling_std}\n    Current Spread: {y[-1]}\n    ")

        return res


    #############################################################################
    #############################################################################
    ################    EXECUTE ORDERS    #######################################
    #############################################################################
    #############################################################################

    ##### RESIN ORDERS #####

    def execute_resin_orders(self, order_depth, state):
        acceptable_price = 10000 # self.acceptable_prices["RAINFOREST_RESIN"]
        limit = self.POSITION_LIMITS["RAINFOREST_RESIN"]
        orders: List[Order] = []
        current_position = state.position.get("RAINFOREST_RESIN", 0)

        sorted_asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
        sorted_bids = sorted(order_depth.buy_orders.items(), key=lambda x: -x[0])

        self.logger_print("RAINFOREST_RESIN", acceptable_price, 0, current_position, limit, [sorted_asks[0], sorted_bids[0]])

        ### BUYING ###
        if len(order_depth.sell_orders) != 0:
            # Sort asks in ascending order (cheapest first)
            
            for ask_price, ask_qty in sorted_asks:
                if ask_price <= acceptable_price:
                    # Maximum we can buy without breaching the long limit:
                    room_to_buy = self.dyn_limit_resin(abs(ask_price - acceptable_price)) - current_position
                    # ask_qty is negative, so take the absolute value.
                    order_qty = min(-ask_qty, room_to_buy)
                    if order_qty > 0:
                        logger.print("BUYING RESIN", order_qty, "x", ask_price)
                        orders.append(Order("RAINFOREST_RESIN", ask_price, order_qty))
                        current_position += order_qty  # Update simulated position.
                        # Stop if limit reached.
                        if current_position >= limit:
                            break

        ### SELLING ###
        if len(order_depth.buy_orders) != 0:
            # Sort bids in descending order (highest bid first)
            
            for bid_price, bid_qty in sorted_bids:
                if bid_price > acceptable_price:
                    # Maximum we can sell without breaching the short limit:
                    room_to_sell = self.dyn_limit_resin(abs(bid_price - acceptable_price)) + current_position  # current_position is positive if long, negative if short.
                    order_qty = min(bid_qty, room_to_sell)
                    if order_qty > 0:
                        logger.print("SELLING RESIN", order_qty, "x", bid_price)
                        orders.append(Order("RAINFOREST_RESIN", bid_price, -order_qty))
                        current_position -= order_qty  # Update simulated position.
                        if current_position <= -limit:
                            break

        ### PURE OU MARKET MAKING ###
        ### HOW THIS WORKS: IF THE MARKET PRICE IS LOWER THAN 10K, WE BUY AT PRICE - 1
        ###     AND SELL AT 10001. IF THE MARKET PRICE IS HIGHER THAN 10K, WE BUY AT 
        ###     9999 AND SELL AT PRICE + 1
        market_price = self.acceptable_prices["RAINFOREST_RESIN"]
        space_to_buy = self.dyn_limit_resin(abs(acceptable_price - market_price)) - current_position
        space_to_sell = self.dyn_limit_resin(abs(acceptable_price - market_price)) + current_position

        if market_price < acceptable_price:
            orders.append(Order("RAINFOREST_RESIN", round(market_price)-1, space_to_buy//2))#POSITIVE NUMBER
            orders.append(Order("RAINFOREST_RESIN", acceptable_price+1, -space_to_sell//2))

        else:
            orders.append(Order("RAINFOREST_RESIN", acceptable_price-1, space_to_buy//2))
            orders.append(Order("RAINFOREST_RESIN", round(market_price)+1, -space_to_sell//2 ))
            
        return orders

    ##### KELP ORDERS #####

    def execute_kelp_orders(self, order_depth, state):
        limit = self.POSITION_LIMITS["KELP"]
        orders: List[Order] = []
        current_position = state.position.get("KELP", 0)
        k_price = self.acceptable_prices["KELP"]
        acceptable_price = k_price

        sorted_asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
        sorted_bids = sorted(order_depth.buy_orders.items(), key=lambda x: -x[0])

        # basic bitch market making #
        res_price = round(self.acceptable_prices["KELP"] - (0.05 * current_position))
        spread = 1

        space_to_buy = self.dyn_limit_resin(abs(acceptable_price - res_price)) - current_position
        space_to_sell = self.dyn_limit_resin(abs(acceptable_price - res_price)) + current_position

        orders.append(Order("KELP", math.ceil(res_price + spread), -space_to_sell))
        orders.append(Order("KELP", math.ceil(res_price - spread), space_to_buy))


        return orders

    ##### INK ORDERS #####

    def execute_ink_orders(self, order_depth, state):
        limit = self.POSITION_LIMITS["SQUID_INK"]
        orders: List[Order] = []
        current_position = state.position.get("SQUID_INK", 0)
        k_price = self.acceptable_prices["KELP"]
        i_price = self.acceptable_prices["SQUID_INK"]

        self.ink_price_hist.append(i_price)
        ink_prices = len(self.ink_price_hist)

        acceptable_price = 0.

        if ink_prices < 40:

            time_weight = np.array([x for x in range(ink_prices+1)[1:]])

            acceptable_price = (2 * np.dot(time_weight, np.array(self.ink_price_hist))) / (ink_prices * (ink_prices + 1))

            self.EMA=acceptable_price

        else:

            acceptable_price = (i_price * 0.02) + (0.98 * self.EMA)

        self.ink_vol_sum += (i_price - acceptable_price) ** 2
        self.ink_vol_ctr += 1

        self.ink_vol = math.sqrt(self.ink_vol_sum / self.ink_vol_ctr)

        sorted_asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
        sorted_bids = sorted(order_depth.buy_orders.items(), key=lambda x: -x[0])

        if abs(acceptable_price - i_price) > self.ink_vol * 2:

            ### BUYING ###
            if len(order_depth.sell_orders) != 0:
                # Sort asks in ascending order (cheapest first)

                for ask_price, ask_qty in sorted_asks:
                    if ask_price <= acceptable_price:
                        # Maximum we can buy without breaching the long limit:
                        room_to_buy = self.dyn_limit_ink(abs(acceptable_price - ask_price) - 2 * self.ink_vol) - current_position
                        # ask_qty is negative, so take the absolute value.
                        order_qty = min(-ask_qty, room_to_buy)
                        if order_qty > 0:
                            logger.print("BUYING INK", order_qty, "x", ask_price)
                            orders.append(Order("SQUID_INK", ask_price, order_qty))
                            current_position += order_qty  # Update simulated position.
                            # Stop if limit reached.
                            if current_position >= limit:
                                break

            ### SELLING ###
            if len(order_depth.buy_orders) != 0:
                # Sort bids in descending order (highest bid first)

                for bid_price, bid_qty in sorted_bids:
                    if bid_price > acceptable_price :
                        # Maximum we can sell without breaching the short limit:
                        room_to_sell = self.dyn_limit_ink(abs(bid_price - acceptable_price) - 2 * self.ink_vol) + current_position  # current_position is positive if long, negative if short.
                        order_qty = min(bid_qty, room_to_sell)
                        if order_qty > 0:
                            logger.print("SELLING INK", acceptable_price, bid_price, self.ink_vol, order_qty)
                            orders.append(Order("SQUID_INK", bid_price, -order_qty))
                            current_position -= order_qty  # Update simulated position.
                            if current_position <= -limit:
                                break

        # NEUTRALIZE INK #
        if abs(acceptable_price - i_price) < max(2 * self.ink_vol, 10):
            if current_position > 0: # If current position is >0, we want to sell to neutralize market position
                spread = 0 # 2 - math.floor(current_position/25)
                orders.append(Order("SQUID_INK", math.ceil(i_price + spread), -current_position))
                logger.print(f"REQUESTING TO SELL {current_position} MORE INK TO NEUTRALIZE MARKET POSITION AT PRICE {i_price + spread}")

            elif current_position < 0: # If current position is <0, we want to buy to neutralize market position
                spread = 0 # 2 + math.ceil(current_position/25)
                orders.append(Order("SQUID_INK", math.floor(i_price - spread), -current_position))
                logger.print(f"REQUESTING TO BUY {-current_position} MORE INK TO NEUTRALIZE MARKET POSITION AT PRICE {acceptable_price - spread}")


        return orders
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

    #### Options Orders ####
    def execute_options(self, state, order_depths):

        res = {}
        
        s1 = self.options_signal1(state, order_depths)

        products = ["VOLCANIC_ROCK_VOUCHER_9500", 
                    "VOLCANIC_ROCK_VOUCHER_9750",
                    "VOLCANIC_ROCK_VOUCHER_10000",
                    "VOLCANIC_ROCK_VOUCHER_10250",
                    "VOLCANIC_ROCK_VOUCHER_10500"]

        quantities = [state.position.get("VOLCANIC_ROCK_VOUCHER_9500", 0),
                      state.position.get("VOLCANIC_ROCK_VOUCHER_9750", 0),
                      state.position.get("VOLCANIC_ROCK_VOUCHER_10000", 0),
                      state.position.get("VOLCANIC_ROCK_VOUCHER_10250", 0),
                      state.position.get("VOLCANIC_ROCK_VOUCHER_10500", 0)]

        deltas = list()
        S = self.mid_price(order_depths["VOLCANIC_ROCK"])
        T = 7 - state.timestamp/1000000

        ctr = 0
        
        for x in products:
            signal = 1 * s1.get(x, 0) # + 0 * s2.get(x, 0)
            target_position = signal * self.POSITION_LIMITS[x]
            qty = int(target_position - state.position.get(x, 0))
            # quantities[ctr] += qty
            deltas.append(self.delta(S, ctr*250 + 9500, T))

            if signal > 0:
                if state.position.get(x, 0) < target_position:
                    price = 0
                    if order_depths[x].sell_orders:
                        price = min(order_depths[x].sell_orders.keys())
                    elif self.options_mid_price_hist.get(x):
                        price = math.floor(self.options_mid_price_hist.get(x, 0)[-1])
                    res[x] = Order(x, price, qty)
                    
            elif signal < 0:
                if state.position.get(x, 0) > target_position:
                    price = 999999
                    if order_depths[x].buy_orders:
                        price = max(order_depths[x].buy_orders.keys())
                    elif self.options_mid_price_hist.get(x):
                        price = math.ceil(self.options_mid_price_hist.get(x, 0)[-1])
                    res[x] = Order(x, price, qty)
            ctr += 1


        # DELTA HEDGE #
        
        target_position = -np.dot(quantities, deltas) // 2
        amount_to_trade = (target_position - state.position.get("VOLCANIC_ROCK", 0))//5

        # Delta is positive, so we should short the underlying
        if target_position > 160:
            room_to_buy = self.POSITION_LIMITS["VOLCANIC_ROCK"] - state.position.get("VOLCANIC_ROCK", 0)
            price = 0
            if order_depths["VOLCANIC_ROCK"].sell_orders:
                price = min(order_depths["VOLCANIC_ROCK"].sell_orders.keys())
            elif self.options_mid_price_hist.get(x):
                price = math.floor(self.options_mid_price_hist.get(x, 0)[-1])
            res["VOLCANIC_ROCK"] = Order('VOLCANIC_ROCK', price, int(min(room_to_buy, amount_to_trade)))
            logger.print(f"ortfolio Delta: {-target_position}, Change in position: {int(amount_to_trade)}, max amount to short: {room_to_buy}")
        elif target_position < -160:
            room_to_sell = -self.POSITION_LIMITS["VOLCANIC_ROCK"] - state.position.get("VOLCANIC_ROCK", 0)
            price = 0
            if order_depths["VOLCANIC_ROCK"].buy_orders:
                price = max(order_depths["VOLCANIC_ROCK"].buy_orders.keys())
            elif self.options_mid_price_hist.get(x):
                price = math.ceil(self.options_mid_price_hist.get(x, 0)[-1])
            res["VOLCANIC_ROCK"] = Order('VOLCANIC_ROCK', price, int(max(room_to_sell, amount_to_trade)))
            logger.print(f"Portfolio Delta: {-target_position}, Change in position: {int(amount_to_trade)}, max amount to long: {room_to_sell}")

        return res
        
    




    #############################################################################
    #############################################################################
    ################    MAIN FUNCTION TO RUN    #################################
    #############################################################################
    #############################################################################

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input,
        # and outputs a list of orders to be sent.
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))
        result = {}
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)


        self.acceptable_prices = self.vwap(state)

        ### MAKE ORDERS ###

        result["RAINFOREST_RESIN"] = self.execute_resin_orders(state.order_depths["RAINFOREST_RESIN"], state)
        result["KELP"] = self.execute_kelp_orders(state.order_depths["KELP"], state)
        result["SQUID_INK"] = self.execute_ink_orders(state.order_depths["SQUID_INK"], state)

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
            result[Product.JAMS] = []
            result[Product.DJEMBES] =  basket1_orders.get(Product.DJEMBES, [])
            result[Product.PICNIC_BASKET1] = basket1_orders.get(Product.PICNIC_BASKET1, [])

        if basket2_orders:
            # Use += if CROISSANTS/JAMS were already in basket1_orders
            result[Product.CROISSANTS] = [] #result.get(Product.CROISSANTS, []) + basket2_orders.get(Product.CROISSANTS, [])
            result[Product.JAMS] = result.get(Product.JAMS, []) + basket2_orders.get(Product.JAMS, [])
            result[Product.PICNIC_BASKET2] = basket2_orders.get(Product.PICNIC_BASKET2, [])

        for x,y in self.execute_options(state, state.order_depths).items():
            logger.print(x, y)
            result[x] = [y]


        traderData = jsonpickle.encode(traderObject)

        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
