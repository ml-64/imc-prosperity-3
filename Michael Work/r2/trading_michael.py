import json
import numpy as np
import pandas as pd
import math
from typing import Any, Dict, List
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


#######################################################
#######################################################
###########                             ###############
###########          BIG ONE            ###############
###########                             ###############
#######################################################
#######################################################


class Trader:

    #############################################################################
    #############################################################################
    ################          Class variables       #############################
    #############################################################################
    #############################################################################
    
    POSITION_LIMITS = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50,
        "SQUID_INK": 50
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
        else: return 30 + 10 * price_diff
        # return math.ceil(25+(25* (1-(0.5 ** float(price_diff)))))

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
            orders.append(Order("RAINFOREST_RESIN", market_price-1, space_to_buy//2))#POSITIVE NUMBER
            orders.append(Order("RAINFOREST_RESIN", acceptable_price+1, -space_to_sell//2))

        else:
            orders.append(Order("RAINFOREST_RESIN", acceptable_price-1, space_to_buy//2))
            orders.append(Order("RAINFOREST_RESIN", market_price+1, -space_to_sell//2 ))
            

        # ### EXCESS OFFLOADING VIA MARKET MAKING ###
        # if current_position > 0: # If current position is >0, we want to sell to neutralize market position
        #     spread = 5 - math.floor(current_position/10)
        #     orders.append(Order("RAINFOREST_RESIN", acceptable_price + spread, -current_position))
        #     logger.print(f"REQUESTING TO SELL {current_position} MORE RAINFOREST RESIN TO NEUTRALIZE MARKET POSITION AT PRICE {acceptable_price + spread}")
            
        # elif current_position < 0: # If current position is <0, we want to buy to neutralize market position
        #     spread = 5 + math.ceil(current_position/10)
        #     orders.append(Order("RAINFOREST_RESIN", acceptable_price - spread, -current_position))
        #     logger.print(f"REQUESTING TO BUY {-current_position} MORE RAINFOREST RESIN TO NEUTRALIZE MARKET POSITION AT PRICE {acceptable_price - spread}")
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

        ### BUYING ###
        if len(order_depth.sell_orders) != 0:
            # Sort asks in ascending order (cheapest first)
            
            for ask_price, ask_qty in sorted_asks:
                if ask_price <= acceptable_price :
                    # Maximum we can buy without breaching the long limit:
                    room_to_buy = 50 - current_position
                    # ask_qty is negative, so take the absolute value.
                    order_qty = min(-ask_qty, room_to_buy)
                    if order_qty > 0:
                        logger.print("BUYING KELP", order_qty, "x", ask_price)
                        orders.append(Order("KELP", ask_price, order_qty))
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
                    room_to_sell = 50 + current_position  # current_position is positive if long, negative if short.
                    order_qty = min(bid_qty, room_to_sell)
                    if order_qty > 0:
                        logger.print("SELLING KELP", order_qty, "x", bid_price)
                        orders.append(Order("KELP", bid_price, -order_qty))
                        current_position -= order_qty  # Update simulated position.
                        if current_position <= -limit:
                            break

        ### STOIKOV MARKET MAKING ALGORITHM ###

        res_price = round(self.acceptable_prices["KELP"] - (0.05 * current_position))
        spread = 3

        orders.append(Order("KELP", math.ceil(res_price + spread), -10))
        orders.append(Order("KELP", math.ceil(res_price - spread), 10))
        


        # ### EXCESS OFFLOADING VIA MARKET MAKING ###
        # if current_position > 0: # If current position is >0, we want to sell to neutralize market position
        #     spread = 5 - math.floor(current_position/10)
        #     orders.append(Order("KELP", math.ceil(acceptable_price + spread), -current_position))
        #     logger.print(f"REQUESTING TO SELL {current_position} MORE RAINFOREST RESIN TO NEUTRALIZE MARKET POSITION AT PRICE {acceptable_price + spread}")
            
        # elif current_position < 0: # If current position is <0, we want to buy to neutralize market position
        #     spread = 5 + math.floor(current_position/10)
        #     orders.append(Order("KELP", round(acceptable_price - spread), -current_position))
        #     logger.print(f"REQUESTING TO BUY {-current_position} MORE RAINFOREST RESIN TO NEUTRALIZE MARKET POSITION AT PRICE {acceptable_price - spread}")

        # ### PAIR TRADING METHOD ###
        # diff = self.pair_limit(((i_price//4) + 1500) - k_price)
        
        # if abs(diff) > 50:
        #     diff = 50 if diff > 50 else -50
        # desired_position = diff - current_position

        # if desired_position > 0:
        #     for x in sorted_asks:
        #         orders.append(Order("KELP", x[0], min(desired_position, -x[1])))
        #         desired_position -= min(desired_position, -x[1])
        # else:
        #     for x in sorted_bids:
        #         orders.append(Order("KELP", x[0], -min(-desired_position, x[1])))
        #         desired_position += min(-desired_position, x[1])

        # logger.print(f"INK INFO: \n    Price of Kelp: {k_price}\n    Adjusted Price of Ink: {i_price}\n    Current Position: {current_position}\n    Desired Position: {diff}\n    Remaining Desired Orders: {desired_position}") 
        
        

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
                            
        if abs(acceptable_price - i_price) < max(2 * self.ink_vol, 10):
            if current_position > 0: # If current position is >0, we want to sell to neutralize market position
                spread = 0 # 2 - math.floor(current_position/25)
                orders.append(Order("SQUID_INK", math.ceil(i_price + spread), -current_position))
                logger.print(f"REQUESTING TO SELL {current_position} MORE INK TO NEUTRALIZE MARKET POSITION AT PRICE {i_price + spread}")
                
            elif current_position < 0: # If current position is <0, we want to buy to neutralize market position
                spread = 0 # 2 + math.ceil(current_position/25)
                orders.append(Order("SQUID_INK", math.floor(i_price - spread), -current_position))
                logger.print(f"REQUESTING TO BUY {-current_position} MORE INK TO NEUTRALIZE MARKET POSITION AT PRICE {acceptable_price - spread}")

        ### PAIR TRADING METHOD ###
        # diff = self.pair_limit(k_price - ((i_price//4) + 1500))
        # if abs(diff) > 50:
        #     diff = 50 if diff > 50 else -50
        
        # desired_position = diff - current_position

        # if desired_position > 0:
        #     for x in sorted_asks:
        #         orders.append(Order("SQUID_INK", x[0], min(desired_position, -x[1])))
        #         desired_position -= min(desired_position, -x[1])
        # else:
        #     for x in sorted_bids:
        #         orders.append(Order("SQUID_INK", x[0], -min(-desired_position, x[1])))
        #         desired_position += min(-desired_position, x[1])

        # logger.print(f"INK INFO: \n    Price of Kelp: {k_price}\n    Adjusted Price of Ink: {i_price}\n    Current Position: {current_position}\n    Desired Position: {diff}\n    Remaining Desired Orders: {desired_position}") 

        return orders

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

        self.acceptable_prices = self.vwap(state)
        
        ### MAKE ORDERS ###

        # result["RAINFOREST_RESIN"] = self.execute_resin_orders(state.order_depths["RAINFOREST_RESIN"], state)
        result["KELP"] = self.execute_kelp_orders(state.order_depths["KELP"], state)
        # result["SQUID_INK"] = self.execute_ink_orders(state.order_depths["SQUID_INK"], state)
        result["RAINFOREST_RESIN"] = []
        # result["KELP"] = []
        result["SQUID_INK"] = []

        traderData = "SAMPLE"  # Pass back state if needed.
        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
