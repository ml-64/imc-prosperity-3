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
    SPREAD_3 = "SPREAD_3"


PARAMS = {
    Product.SPREAD: {
        "default_spread_mean": 70.0449,
        "default_spread_std": 78.66710516685775,
        "spread_std_window": 50,
        "zscore_threshold": 2.711135,
        "target_position": 58,
        "relative": 0,
        },
  Product.SPREAD_B2: {
        "default_spread_mean": 58.0801,
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
        "DJEMBES": 60,
        "VOLCANIC_ROCK": 400,
        "VOLCANIC_ROCK_VOUCHER_9500": 200,
        "VOLCANIC_ROCK_VOUCHER_9750": 200,
        "VOLCANIC_ROCK_VOUCHER_10000": 200,
        "VOLCANIC_ROCK_VOUCHER_10250": 200,
        "VOLCANIC_ROCK_VOUCHER_10500": 200
    }

    options_spread_hist = {}
    options_mid_price_hist = {}
    options_vol_spread_hist = {}

    default_options_vol = {
        "VOLCANIC_ROCK_VOUCHER_9500": 9.250213706500483,
        "VOLCANIC_ROCK_VOUCHER_9750": 16.113672371581206,
        "VOLCANIC_ROCK_VOUCHER_10000": 22.150063341406153,
        "VOLCANIC_ROCK_VOUCHER_10250": 13.422148772131159,
        "VOLCANIC_ROCK_VOUCHER_10500": 14.443010012913295
    }

    default_options_params = [(-12.117555633410266, -0.9539866769309805, -0.007561177345686292, 0.0037386720664748813),
                              (20.62342988126337, -1.0837293167631605, -0.0034601547733240172, 0.0019571880952035034),
                              (-0.13371995437950882, -0.09542869910941022, 0.008028418830807063, 0.00034151831911412365),
                              (0.5886483608831591, -0.015597647533150974, 0.008630530487557133, 8.768514282620155e-05),
                              (1.1627880648135522, 0.03505665677218081, 0.008513613574124314, 8.78160932482336e-05)]

    def logger_print(self, product, width, pred, current_position, limit, cheapest):
        logger.print(f"{product} - Fair Bid: {pred-width}, Current position: {current_position}, Limit: {limit}, Cheapest Ask: {cheapest[0]}\n")
        logger.print(f"{product} - Fair Ask: {pred+width}, Current position: {current_position}, Limit: {limit}, Cheapest Bid: {cheapest[1]}\n")

    #############################################################################
    #############################################################################
    ################    PRICING HELPER FUNCTIONS    #############################
    #############################################################################
    #############################################################################

    # VOLUME WEIGHTED AVERAGE PRICING
    ### The Stanford 2024 team noticed that the true "mid price" tends to
    ### be determined by an automated market maker who sets a high volume
    ### on all their orders. This helper function takes the order depth and
    ### takes the average value, weighted in favor of orders with higher volume

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

    def black_scholes_implied(self, S, K, T, sigma):
        """Black-Scholes price for a European call option"""
        d1 = (np.log(S / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * NormalDist().cdf(d1) - K * NormalDist().cdf(d2)

    def vega(self, S, K, T, sigma):
        """Derivative of BS price with respect to volatility (vega)"""
        d1 = (np.log(S / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * NormalDist().pdf(d1) * np.sqrt(T)
    
    def implied_vol_newton(self, price, S, K, timestamp, sigma0=0.2, tol=1e-6, max_iter=100):
        T = 7 - timestamp/1000000
        sigma = sigma0
        try: 
            for i in range(max_iter):
                implied = self.black_scholes_implied(S, K, T, sigma)
                diff = implied-price
                if abs(diff) < tol:
                    return sigma
                v = self.vega(S, K, T, sigma)
                if v < 1e-6:
                    break
                change = diff / v
                if sigma - change < 1e-6:
                    return sigma
                else:
                    sigma -= diff / v
            return sigma
        except OverflowError:
            return 1e-6

    def m_t(self, S, K, timestamp):
        T = 7 - (timestamp / 1000000)
        return (np.log(K / S)) / np.sqrt(T)

    def V_t(self, weights, m_t):
        var = np.array([m_t**2, m_t, 1])
        return np.dot(weights, var)

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

    def options_signal2(self, state, order_depths):
        
        S = self.mid_price(order_depths["VOLCANIC_ROCK"])
        K = np.array([9500, 9750, 10000, 10250, 10500])        
        timestamp = state.timestamp

        # FIND TRUE IV #
        m_t = [self.m_t(S, K[x], timestamp) for x in range(5)] 
        true_IV = [self.V_t(self.default_options_params[x][:3], m_t[x]) for x in range(5)]

        # FIND CURRENT IV #
        mid_prices = np.array([self.mid_price(order_depths["VOLCANIC_ROCK_VOUCHER_9500"]),
                               self.mid_price(order_depths["VOLCANIC_ROCK_VOUCHER_9750"]),
                               self.mid_price(order_depths["VOLCANIC_ROCK_VOUCHER_10000"]),
                               self.mid_price(order_depths["VOLCANIC_ROCK_VOUCHER_10250"]),
                               self.mid_price(order_depths["VOLCANIC_ROCK_VOUCHER_10500"])])
        
        cur_IV = [self.implied_vol_newton(mid_prices[x], S, K[x], timestamp, true_IV[x]) for x in range(5)]

        z_score = (np.array(true_IV) - np.array(cur_IV))/ np.array([self.default_options_params[x][3] for x in range(5)])

        # TRADE IF Z SCORE HIGH ENOUGH # 
        res = {}
        
        for x in range(5):

            prod = f"VOLCANIC_ROCK_VOUCHER_{str(250*x+9500)}"

            if self.options_vol_spread_hist.get(prod):
                cur = self.options_vol_spread_hist.get(prod)
                cur.append(true_IV[x] - cur_IV[x])
                if len(cur) > 100:
                    cur.pop(0)
                self.options_vol_spread_hist[prod] = cur
            else:
                self.options_vol_spread_hist[prod] = [true_IV[x] - cur_IV[x]]
             
            rolling_STD = np.std(self.options_vol_spread_hist.get(prod))

            if rolling_STD < 1e-6:
                rolling_STD = self.default_options_params[x][3]
            
            z = (true_IV[x] - cur_IV[x])/rolling_STD
            
            logger.print(f"Z SCORE for {prod}: {z}")
            if z > 2:
                res[prod] = -1
            elif z < -2:
                res[prod] = 1
            else:
                res[prod] = 0
        
        return res
        

    def execute_options(self, state, order_depths):

        res = {}
        
        s1 = self.options_signal1(state, order_depths)
        s2 = self.options_signal2(state, order_depths)

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
        T = 5 - state.timestamp/1000000

        ctr = 0
        
        for x in products:
            signal = 0.7 * s1.get(x, 0) + 0.3 * s2.get(x, 0)
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
        
        target_position = -np.dot(quantities, deltas)//2
        amount_to_trade = (target_position - state.position.get("VOLCANIC_ROCK", 0)) //5

        # Delta is positive, so we should short the underlying
        if target_position > 50:
            room_to_buy = self.POSITION_LIMITS["VOLCANIC_ROCK"] - state.position.get("VOLCANIC_ROCK", 0)
            price = 0
            if order_depths["VOLCANIC_ROCK"].sell_orders:
                price = min(order_depths["VOLCANIC_ROCK"].sell_orders.keys())
            elif self.options_mid_price_hist.get(x):
                price = math.flootfr(self.options_mid_price_hist.get(x, 0)[-1])
            res["VOLCANIC_ROCK"] = Order('VOLCANIC_ROCK', price, int(min(room_to_buy, amount_to_trade)))
            logger.print(f"Portfolio Delta: {-target_position}, Change in position: {int(amount_to_trade)}, max amount to short: {room_to_buy}")
        elif target_position < 50:
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

        ### MAKE ORDERS ###

        result["RAINFOREST_RESIN"] = []
        result["KELP"] = []
        result["SQUID_INK"] = []
        result["JAMS"] = []
        result["CROISSANTS"] = []
        result["PICNIC_BASKET2"] = []
        result["PICNIC_BASKET1"] = []
        result["DJEMBES"] = []

        for x,y in self.execute_options(state, state.order_depths).items():
            logger.print(x, y)
            result[x] = [y]

        traderData = jsonpickle.encode(traderObject)

        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
