import json
import numpy as np
import pandas as pd
import math
from typing import Any, Dict, List
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import string

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
    SPREAD_3 = "SPREAD_3"


PARAMS = {
    Product.SPREAD: {
        "default_spread_mean": 58,
        "default_spread_std": 78.66710516685775,
        "spread_std_window": 50,
        "zscore_threshold": 2.711135,
        "target_position": 58,
        "relative": 0,
        },
  Product.SPREAD_B2: {
        "default_spread_mean": 30.33,
        "default_spread_std": 51.88543913150766,
        "spread_std_window": 50,
        "zscore_threshold": 2.604117,
        "target_position": 98,
        "relative": 0,
    },
   Product.SPREAD_3:{
        "default_spread_mean": -17.07525,
        "default_spread_std": 73.41289554813106,
        "spread_std_window": 50,
        "zscore_threshold": 2.699572,
        "target_position": 58//2,
   }
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
        "DJEMBES": 60
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



    def create_spread_orders_basket1(
              self,
              order_depths: Dict[str, OrderDepth],
              product: Product,
              basket_position: int,
              spread_data: Dict[str, Any],
          ):
              if Product.PICNIC_BASKET1 not in order_depths.keys():
                  return None

              basket_order_depth = order_depths[Product.PICNIC_BASKET1]
              synthetic_order_depth = self.synthetic_basket1_order_depth(order_depths)
              basket_vwap = self.vwap_single(basket_order_depth)
              synthetic_vwap = self.vwap_single(synthetic_order_depth)
              spread = basket_vwap - synthetic_vwap
              spread_data["spread_history"].append(spread)

              if (
                  len(spread_data["spread_history"])
                  < self.params[Product.SPREAD]["spread_std_window"]
              ):
                  return None
              elif len(spread_data["spread_history"]) > self.params[Product.SPREAD]["spread_std_window"]:
                  spread_data["spread_history"].pop(0)

              spread_std = np.std(spread_data["spread_history"])

              zscore = (
                  spread - self.params[Product.SPREAD]["default_spread_mean"]
              ) / spread_std

              if zscore >= self.params[Product.SPREAD]["zscore_threshold"]:
                  if basket_position != -self.params[Product.SPREAD]["target_position"]:
                      return self.execute_PICNIC_BASKET1_orders(
                          -self.params[Product.SPREAD]["target_position"],
                          basket_position,
                          order_depths,
                      )

              if zscore <= -self.params[Product.SPREAD]["zscore_threshold"]:
                  if basket_position != self.params[Product.SPREAD]["target_position"]:
                      return self.execute_PICNIC_BASKET1_orders(
                          self.params[Product.SPREAD]["target_position"],
                          basket_position,
                          order_depths,
                      )

              spread_data["prev_zscore"] = zscore
              return None



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




    def create_spread_orders_basket2(
              self,
              order_depths: Dict[str, OrderDepth],
              product: Product,
              basket_position: int,
              spread_data: Dict[str, Any],
          ):
              if Product.PICNIC_BASKET2 not in order_depths.keys():
                  return None

              basket_order_depth = order_depths[Product.PICNIC_BASKET2]
              synthetic_order_depth = self.synthetic_basket2_order_depth(order_depths)
              basket_vwap = self.vwap_single(basket_order_depth)
              synthetic_vwap = self.vwap_single(synthetic_order_depth)
              spread = basket_vwap - synthetic_vwap
              spread_data["spread_history"].append(spread)

              if (
                  len(spread_data["spread_history"])
                  < self.params[Product.SPREAD_B2]["spread_std_window"]
              ):
                  return None
              elif len(spread_data["spread_history"]) > self.params[Product.SPREAD_B2]["spread_std_window"]:
                  spread_data["spread_history"].pop(0)

              spread_std = np.std(spread_data["spread_history"])

              zscore = (
                  spread - self.params[Product.SPREAD_B2]["default_spread_mean"]
              ) / spread_std

              if zscore >= self.params[Product.SPREAD_B2]["zscore_threshold"]:
                  if basket_position != -self.params[Product.SPREAD_B2]["target_position"]:
                      return self.execute_PICNIC_BASKET2_orders(
                          -self.params[Product.SPREAD_B2]["target_position"],
                          basket_position,
                          order_depths,
                      )

              if zscore <= -self.params[Product.SPREAD_B2]["zscore_threshold"]:
                  if basket_position != self.params[Product.SPREAD_B2]["target_position"]:
                      return self.execute_PICNIC_BASKET2_orders(
                          self.params[Product.SPREAD_B2]["target_position"],
                          basket_position,
                          order_depths,
                      )

              spread_data["prev_zscore"] = zscore
              return None

    #for later for michael to edit 
    def create_spread_orders_interbasket(self, order_depths: Dict[str, OrderDepth], spread_data) -> float:
            if Product.PICNIC_BASKET1 not in order_depths.keys():
                  return None

            if Product.PICNIC_BASKET2 not in order_depths.keys():
                  return None

            basket1_order_depth = order_depths[Product.PICNIC_BASKET1]
            basket1_vwap = self.vwap_single(basket1_order_depth)
              
            basket2_order_depth = order_depths[Product.PICNIC_BASKET2]
            basket2_vwap = self.vwap_single(basket2_order_depth)
            
            djembe_order_depth = order_depths[Product.DJEMBES]
            djembe_vwap = self.vwap_single(djembe_order_depth)

            spread = basket1_vwap - djembe_vwap - 1.5 * basket2_vwap
            spread_data["spread_history"].append(spread)

            if (
                  len(spread_data["spread_history"])
                  < self.params[Product.SPREAD_3]["spread_std_window"]
              ):
                  return None
            elif len(spread_data["spread_history"]) > self.params[Product.SPREAD_3]["spread_std_window"]:
                  spread_data["spread_history"].pop(0)
    
            spread_std = np.std(spread_data["spread_history"])
    
            zscore = (
                  spread - self.params[Product.SPREAD_3]["default_spread_mean"]
            ) / spread_std

            
            if zscore >= self.params[Product.SPREAD_3]["zscore_threshold"]:
                  if basket_position != -self.params[Product.SPREAD_3]["target_position"]:
                      return self.execute_dual_basket_trade(
                          -self.params[Product.SPREAD_3]["target_position"],
                          basket_position,
                          order_depths,
                      )

            if zscore <= -self.params[Product.SPREAD_3]["zscore_threshold"]:
                  if basket_position != self.params[Product.SPREAD_3]["target_position"]:
                      return self.execute_dual_basket_trade(
                          self.params[Product.SPREAD_3]["target_position"],
                          basket_position,
                          order_depths,
                      )

            spread_data["prev_zscore"] = zscore
            return None
 

 

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

    ### dual basket trade ###
    def execute_dual_basket_trade(self, target_pos, current_pos, order_depths):
      basket1_orders = []
      basket2_orders = []
      djembe_orders = []

      # determine direction
      qty = target_pos - current_pos
      b1_side = 1 if target_pos > current_pos else -1
      b2_side = -1.5 * b1_side
      djembe_side = -1 * b1_side

      # get prices
      b1_price = min(order_depths[Product.PICNIC_BASKET1].sell_orders.keys()) if target_pos > current_pos else \
                max(order_depths[Product.PICNIC_BASKET1].buy_orders.keys())
      b2_price = min(order_depths[Product.PICNIC_BASKET2].sell_orders.keys()) if target_pos < current_pos else \
                max(order_depths[Product.PICNIC_BASKET2].buy_orders.keys())
      djembe_price = min(order_depths[Product.DJEMBES].sell_orders.keys()) if target_pos < current_pos else \
                    max(order_depths[Product.DJEMBES].buy_orders.keys())

      # create orders
      basket1_orders.append(Order(Product.PICNIC_BASKET1, b1_price, b1_side * qty))
      basket2_orders.append(Order(Product.PICNIC_BASKET2, b2_price, int(b2_side * qty)))
      djembe_orders.append(Order(Product.DJEMBES, djembe_price, djembe_side * qty))

      return {
          Product.PICNIC_BASKET1: basket1_orders,
          Product.PICNIC_BASKET2: basket2_orders,
          Product.DJEMBES: djembe_orders,
      }

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

        # if Product.SPREAD_3 not in traderObject:
        #     traderObject[Product.SPREAD_3] = {
        #         "spread_history": [],
        #         "prev_zscore": 0,
        #     }
        # basket_targets = self.create_spread_orders_interbasket(
        #     state.order_depths,
        #     traderObject[Product.SPREAD_3],
        #     )

        #basket 1 #
        if Product.SPREAD not in traderObject:
                traderObject[Product.SPREAD] = {
                    "spread_history": [],
                    "prev_zscore": 0,
                }

        basket_position_1 = (
            state.position[Product.PICNIC_BASKET1]
            if Product.PICNIC_BASKET1 in state.position
            else 0
            )
        basket1_orders = self.create_spread_orders_basket1(
            state.order_depths,
            Product.PICNIC_BASKET1,
            basket_position_1,
            traderObject[Product.SPREAD],
            )
        if basket1_orders:
            result[Product.CROISSANTS] = basket1_orders[Product.CROISSANTS]
            result[Product.JAMS] = basket1_orders[Product.JAMS]
            result[Product.DJEMBES] = basket1_orders[Product.DJEMBES]
            result[Product.PICNIC_BASKET1] = basket1_orders[Product.PICNIC_BASKET1]
        else:
            result[Product.CROISSANTS] = []
            result[Product.JAMS] = []
            result[Product.DJEMBES] = []
            result[Product.PICNIC_BASKET1] = []

        #basket two#
         # Basket 2 #

        if Product.SPREAD_B2 not in traderObject:
                traderObject[Product.SPREAD_B2] = {
                    "spread_history": [],
                    "prev_zscore": 0,
                }

        basket_position_2 = (
            state.position[Product.PICNIC_BASKET2]
            if Product.PICNIC_BASKET2 in state.position
            else 0
            )
        basket2_orders = self.create_spread_orders_basket2(
            state.order_depths,
            Product.PICNIC_BASKET2,
            basket_position_2,
            traderObject[Product.SPREAD_B2],
            )
            
        if basket2_orders:
            result[Product.CROISSANTS] = result.get(Product.CROISSANTS, [])+basket2_orders[Product.CROISSANTS]
            result[Product.JAMS] = result.get(Product.JAMS, [])+basket2_orders[Product.JAMS]
            result[Product.PICNIC_BASKET2] = result.get(Product.PICNIC_BASKET2, [])+basket2_orders[Product.PICNIC_BASKET2]

        # # dual basket spread
        # if Product.SPREAD_3 not in traderObject:
        #         traderObject[Product.SPREAD_3] = {
        #             "spread_history": [],
        #             "prev_zscore": 0,
        #         }
        # basket_position_3 = (
        #     state.position[Product.PICNIC_BASKET2]
        #     if Product.PICNIC_BASKET2 in state.position
        #     else 0
        #     )

        # basket3_orders = self.create_spread_orders_interbasket(
        #     state.order_depths,
        #     basket_position_3,
        #     traderObject[Product.SPREAD_3]
        #     )
            
        # if basket3_orders:
        #     result[Product.PICNIC_BASKET1] = result.get(Product.PICNIC_BASKET1, [])+basket3_orders[Product.PICNIC_BASKET1]
        #     result[Product.PICNIC_BASKET2] = result.get(Product.PICNIC_BASKET2, [])+basket3_orders[Product.PICNIC_BASKET2]
        #     result[Product.DJEMBES] = result.get(Product.DJEMBES,[])+basket3_orders[Product.DJEMBES]

        # result["RAINFOREST_RESIN"] = []
        # result["KELP"] = []
        # result["SQUID_INK"] = []
        # result["JAMS"] = []
        # result["CROISSANTS"] = []
        # result["PICNIC_BASKET2"] = []
        # result["PICNIC_BASKET1"] = []
        # result["DJEMBES"] = []

        traderData = jsonpickle.encode(traderObject)

        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
