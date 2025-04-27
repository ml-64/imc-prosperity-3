
    ### BLACK SCHOLES ###
    def black_scholes_hist(self, S, K, timestamp):
        T = 3 - (timestamp / 1000000)  # Convert days to years
        sigma = 0.0268
        d1 = (np.log(S / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * NormalDist().cdf(d1) - K * NormalDist().cdf(d2)
        # print(norm.cdf(d2))
        return call_price

    def delta(self, S, K, timestamp):
        T = 3 - (timestamp / 1000000)
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
        T = 3 - timestamp/1000000
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
        T = 3 - (timestamp / 1000000)
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

        default_options_vol = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9.250213706500483,
            "VOLCANIC_ROCK_VOUCHER_9750": 16.113672371581206,
            "VOLCANIC_ROCK_VOUCHER_10000": 22.150063341406153,
            "VOLCANIC_ROCK_VOUCHER_10250": 13.422148772131159,
            "VOLCANIC_ROCK_VOUCHER_10500": 14.443010012913295
        }
        
        for x,y in self.options_spread_hist.items():

            prod = x
            
            # CALCULATE Z-SCORE ON A ROLLING BASIS #
            rolling_mean = np.mean(np.array(y))
            rolling_std = np.std(np.array(y))

            if rolling_std == 0:
                rolling_std = default_options_vol[x]

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
        timestamp = state.timestamp

        ctr = 0
        
        for x in products:
            signal = s1.get(x, 0)
            target_position = signal * self.POSITION_LIMITS[x]
            qty = int(target_position - state.position.get(x, 0))
            # quantities[ctr] += qty
            deltas.append(self.delta(S, ctr*250 + 9500, timestamp))

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
            # logger.print(f"Portfolio Delta: {-target_position}, Change in position: {int(amount_to_trade)}, max amount to short: {room_to_buy}")
        elif target_position < 50:
            room_to_sell = -self.POSITION_LIMITS["VOLCANIC_ROCK"] - state.position.get("VOLCANIC_ROCK", 0)
            price = 0
            if order_depths["VOLCANIC_ROCK"].buy_orders:
                price = max(order_depths["VOLCANIC_ROCK"].buy_orders.keys())
            elif self.options_mid_price_hist.get(x):
                price = math.ceil(self.options_mid_price_hist.get(x, 0)[-1])
            res["VOLCANIC_ROCK"] = Order('VOLCANIC_ROCK', price, int(max(room_to_sell, amount_to_trade)))
            # logger.print(f"Portfolio Delta: {-target_position}, Change in position: {int(amount_to_trade)}, max amount to long: {room_to_sell}")

        return res
