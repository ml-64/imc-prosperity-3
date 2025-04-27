    def execute_CSI_trading(self, state, order_depth, observation):
        self.macaron_price_hist.append(self.mid_price(order_depth))
        if len(self.macaron_price_hist) > 100:
            self.macaron_price_hist.pop(0)
        
        timestamp = state.timestamp

        def CSI_signal(l, r):
            if r > 47.5:
                if l < 47.5: return -1
                else: return 0
            else: return 1
        
        if timestamp%100000==0:
            self.sunlight[0] = observation.sunlightIndex
            return []
        else:
            if timestamp%100000==400:
                self.sunlight[2] = self.sunlight[1]
                self.sunlight[1] = (observation.sunlightIndex - self.sunlight[0])/4

            slope = self.sunlight[1]
    
            l = self.sunlight[0]
            r = l + slope*1000

            res = []

            signal = CSI_signal(l, r)
            self.CSI_signal = signal

            logger.print(f"SIGNAL: {signal, l, r}")

            # NEUTRALIZE EXCESS POSITION #
            if signal == 0 and state.position.get("MAGNIFICENT_MACARONS", 0) > 0:
                price = max(order_depth.buy_orders.values())

                res.append(Order("MAGNIFICENT_MACARONS", price, -state.position.get("MAGNIFICENT_MACARONS", 0)))

            # MARKET TAKING # 
            if timestamp%100000<1400 and timestamp%100000>400:   

                if signal == 1:
                    room_to_trade = self.POSITION_LIMITS["MAGNIFICENT_MACARONS"] - state.position.get("MAGNIFICENT_MACARONS", 0)
                    price = self.macaron_price_hist[-2]
                    if order_depth.sell_orders:
                        price = min(order_depth.sell_orders.keys())
                    res.append(Order("MAGNIFICENT_MACARONS", price, room_to_trade))
                    # logger.print(f"SENDING MACARON BUY ORDER, price: {price}, qty: {room_to_trade}, orders: {order_depth.sell_orders}")
                elif signal == -1:
                    price = self.macaron_price_hist[-2]
                    room_to_trade = -self.POSITION_LIMITS["MAGNIFICENT_MACARONS"] - state.position.get("MAGNIFICENT_MACARONS", 0)
                    if order_depth.buy_orders:
                        price = max(order_depth.buy_orders.keys())

                    logger.print("WTF!")

                    res.append(Order("MAGNIFICENT_MACARONS", price, room_to_trade))

            # STOIKOV MARKET MAKING IF TIME #
            
            return res
