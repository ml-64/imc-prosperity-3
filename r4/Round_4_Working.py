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
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"


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
    
    
    def macarons_implied_bid_ask(
        self,
        observation: ConversionObservation,
    ) -> (float, float):
        return observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1, observation.askPrice + observation.importTariff + observation.transportFees

    def macarons_adap_edge(
        self,
        timestamp: int,
        curr_edge: float,
        position: int,
        traderObject: dict
    ) -> float: 
        if timestamp == 0:
            traderObject["MAGNIFICENT_MACARONS"]["curr_edge"] = self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"]
            return self.params[Product.MAGNIFICENT_MACARONS]["init_make_edge"]

        # Timestamp not 0
        traderObject["MAGNIFICENT_MACARONS"]["volume_history"].append(abs(position))
        if len(traderObject["MAGNIFICENT_MACARONS"]["volume_history"]) > self.params[Product.MAGNIFICENT_MACARONS]["volume_avg_timestamp"]:
            traderObject["MAGNIFICENT_MACARONS"]["volume_history"].pop(0)

        if len(traderObject["MAGNIFICENT_MACARONS"]["volume_history"]) < self.params[Product.MAGNIFICENT_MACARONS]["volume_avg_timestamp"]:
            return curr_edge
        elif not traderObject["MAGNIFICENT_MACARONS"]["optimized"]:
            volume_avg = np.mean(traderObject["MAGNIFICENT_MACARONS"]["volume_history"])

            # Bump up edge if consistently getting lifted full size
            if volume_avg >= self.params[Product.MAGNIFICENT_MACARONS]["volume_bar"]:
                traderObject["MAGNIFICENT_MACARONS"]["volume_history"] = [] # clear volume history if edge changed
                traderObject["MAGNIFICENT_MACARONS"]["curr_edge"] = curr_edge + self.params[Product.MAGNIFICENT_MACARONS]["step_size"]
                return curr_edge + self.params[Product.MAGNIFICENT_MACARONS]["step_size"]

            # Decrement edge if more cash with less edge, included discount
            elif self.params[Product.MAGNIFICENT_MACARONS]["dec_edge_discount"] * self.params[Product.MAGNIFICENT_MACARONS]["volume_bar"] * (curr_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"]) > volume_avg * curr_edge:
                if curr_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"] > self.params[Product.MAGNIFICENT_MACARONS]["min_edge"]:
                    traderObject["MAGNIFICENT_MACARONS"]["volume_history"] = [] # clear volume history if edge changed
                    traderObject["MAGNIFICENT_MACARONS"]["curr_edge"] = curr_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"]
                    traderObject["MAGNIFICENT_MACARONS"]["optimized"] = True
                    return curr_edge - self.params[Product.MAGNIFICENT_MACARONS]["step_size"]
                else:
                    traderObject["MAGNIFICENT_MACARONS"]["curr_edge"] = self.params[Product.MAGNIFICENT_MACARONS]["min_edge"]
                    return self.params[Product.MAGNIFICENT_MACARONS]["min_edge"]

        traderObject["MAGNIFICENT_MACARONS"]["curr_edge"] = curr_edge
        return curr_edge

    def macarons_arb_take(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        adap_edge: float,
        position: int
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]
        buy_order_volume = 0
        sell_order_volume = 0

        implied_bid, implied_ask = self.macarons_implied_bid_ask(observation)                                                                                                                                                                    

        buy_quantity = position_limit - position
        sell_quantity = position_limit + position

        ask = implied_ask + adap_edge

        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 1.6

        if aggressive_ask > implied_ask:
            ask = aggressive_ask

        edge = (ask - implied_ask) * self.params[Product.MAGNIFICENT_MACARONS]["make_probability"]

        for price in sorted(list(order_depth.sell_orders.keys())):
            if price > implied_bid - edge:
                break

            if price < implied_bid - edge:
                quantity = min(abs(order_depth.sell_orders[price]), buy_quantity) # max amount to buy
                if quantity > 0:
                    orders.append(Order(Product.MAGNIFICENT_MACARONS, round(price), quantity))
                    buy_order_volume += quantity

        for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
            if price < implied_ask + edge:
                break

            if price > implied_ask + edge:
                quantity = min(abs(order_depth.buy_orders[price]), sell_quantity) # max amount to sell
                if quantity > 0:
                    orders.append(Order(Product.MAGNIFICENT_MACARONS, round(price), -quantity))
                    sell_order_volume += quantity

        return orders, buy_order_volume, sell_order_volume

    def macarons_arb_clear(
        self,
        position: int
    ) -> int:
        conversions = -position
        return conversions

    def macarons_arb_make(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        position: int,
        edge: float,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]

        # Implied Bid = observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1
        # Implied Ask = observation.askPrice + observation.importTariff + observation.transportFees
        implied_bid, implied_ask = self.macarons_implied_bid_ask(observation)

        bid = implied_bid - edge
        ask = implied_ask + edge

        # ask = foreign_mid - 1.6 best performance so far
        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 1.6 # Aggressive ask

        # don't lose money
        if aggressive_ask >= implied_ask + self.params[Product.MAGNIFICENT_MACARONS]['min_edge']:
            ask = aggressive_ask
            logger.print("AGGRESSIVE")
            logger.print(f"ALGO ASK: {round(ask)}")
            logger.print(f"ALGO BID: {round(bid)}")
        else:
            logger.print(f"ALGO ASK: {round(ask)}")
            logger.print(f"ALGO BID: {round(bid)}")

        filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 40]
        filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 25]

        # If we're not best level, penny until min edge
        if len(filtered_ask) > 0 and ask > filtered_ask[0]:
            if filtered_ask[0] - 1 > implied_ask:
                ask = filtered_ask[0] - 1
            else:
                ask = implied_ask + edge
        if len(filtered_bid) > 0 and  bid < filtered_bid[0]:
            if filtered_bid[0] + 1 < implied_bid:
                bid = filtered_bid[0] + 1
            else:
                bid = implied_bid - edge

        logger.print(f"IMPLIED_BID: {implied_bid}")
        logger.print(f"IMPLIED_ASK: {implied_ask}")
        logger.print(f"FOREIGN ASK: {observation.askPrice}")
        logger.print(f"FOREIGN BID: {observation.bidPrice}")

        best_bid = min(order_depth.buy_orders.keys())
        best_ask = max(order_depth.sell_orders.keys())

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.MAGNIFICENT_MACARONS, round(bid), buy_quantity))  # Buy order

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.MAGNIFICENT_MACARONS, round(ask), -sell_quantity))  # Sell order

        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
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
                macarons_take_orders + macarons_make_orders
            )

        traderData = jsonpickle.encode(traderObject)

        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

