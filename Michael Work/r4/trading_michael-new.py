 # MID PRICE HISTORY FOR MACARONS #
    macaron_price_hist = list()

    def logger_print(self, product, width, pred, current_position, limit, cheapest):
        logger.print(f"{product} - Fair Bid: {pred-width}, Current position: {current_position}, Limit: {limit}, Cheapest Ask: {cheapest[0]}\n")
        logger.print(f"{product} - Fair Ask: {pred+width}, Current position: {current_position}, Limit: {limit}, Cheapest Bid: {cheapest[1]}\n")

    #############################################################################
    #############################################################################
    ################    PRICING HELPER FUNCTIONS    #############################
    #############################################################################
    #############################################################################

    def mid_price(self, order_depth):
        sells = len(order_depth.sell_orders.keys())
        buys = len(order_depth.buy_orders.keys())

        if sells > 0 and buys > 0:
            return (min(order_depth.sell_orders.keys()) + max(order_depth.buy_orders.keys()))/2

        elif sells > 0:
            return min(order_depth.sell_orders.keys())

        else:
            return max(order_depth.buy_orders.keys())
        
    # OK LETS SEE #
    def CSI_trading(self, state, order_depth, observation):
        self.macaron_price_hist.append(self.mid_price(order_depth))
        if len(self.macaron_price_hist) > 100:
            self.macaron_price_hist.pop(0)
        
        timestamp = state.timestamp

        def CSI_signal(l, r):
            if r > 47:
                
                if l < 47: return -1
                else: return 0
            else: 
                
                if 47-r < 0: return 0
                else: return 1
        
        if timestamp%100000==0:
            self.sunlight[0] = observation.sunlightIndex
            return []
        else:
            if timestamp%100000==100:
                self.sunlight[2] = self.sunlight[1]
                self.sunlight[1] = observation.sunlightIndex - self.sunlight[0]

            slope = self.sunlight[1]
    
            l = self.sunlight[0]
            r = l + slope*1000

            res = []

            signal = CSI_signal(l, r)

            # NEUTRALIZE EXCESS POSITION #
            if signal == 0 and state.position.get("MAGNIFICENT_MACARONS", 0) > 0:
                price = max(order_depth.buy_orders.values())

                res.append(Order("MAGNIFICENT_MACARONS"), price, -state.position.get("MAGNIFICENT_MACARONS", 0))

            # MARKET TAKING # 
            if timestamp%100000<1000:   

                if signal == 1:
                    room_to_trade = self.POSITION_LIMITS["MAGNIFICENT_MACARONS"] - state.position.get("MAGNIFICENT_MACARONS", 0)
                    price = self.macaron_price_hist[-2]
                    if order_depth.sell_orders:
                        price = min(order_depth.sell_orders.keys())
                    res.append(Order("MAGNIFICENT_MACARONS", price, room_to_trade))
                    logger.print(f"SENDING MACARON BUY ORDER, price: {price}, qty: {room_to_trade}, orders: {order_depth.sell_orders}")
                elif signal == -1:
                    price = self.macaron_price_hist[-2]
                    room_to_trade = -self.POSITION_LIMITS["MAGNIFICENT_MACARONS"] - state.position.get("MAGNIFICENT_MACARONS", 0)
                    if order_depth.buy_orders:
                        price = max(order_depth.buy_orders.keys())

                    res.append(Order("MAGNIFICENT_MACARONS", price, room_to_trade))

            # STOIKOV MARKET MAKING IF TIME #
            
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
        result["MAGNIFICENT_MACARONS"] = self.CSI_trading(state, state.order_depths.get("MAGNIFICENT_MACARONS", OrderDepth()), state.observations.conversionObservations["MAGNIFICENT_MACARONS"])

        traderData = jsonpickle.encode(traderObject)

        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
