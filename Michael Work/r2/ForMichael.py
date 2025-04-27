       Product.SPREAD.INTER:{
        "default_spread_mean": -17.07525,
        "default_spread_std": 73.41289554813106,
        "spread_std_window": 50,
        "zscore_threshold": 2.699572,
        "target_position": 100,
    }
}
   
    #for later for michael to edit 
    def create_spread_orders_interbasket(self, order_depths: Dict[str, OrderDepth]) -> float:
            if Product.PICNIC_BASKET2 not in order_depths.keys():
                  return None

            basket1_order_depth = order_depths[Product.PICNIC_BASKET1]
            basket_vwap = self.vwap_single(basket1_order_depth)
              
            basket2_order_depth = order_depths[Product.PICNIC_BASKET2]
            basket_vwap = self.vwap_single(basket2_order_depth)
            
            djembe_order_depth = order_depths[Product.DJEMBES]
            djembe_vwap = self.vwap_single(Product.DJEMBES)

            spread = basket1_vwap - djembe_vwap - 1.5 * basket2_vwap
            spread_data["spread_history"].append(spread)

            if (
                  len(spread_data["spread_history"])
                  < self.params[Product.SPREAD_INTER]["spread_std_window"]
              ):
                  return None
            elif len(spread_data["spread_history"]) > self.params[Product.SPREAD_INTER]["spread_std_window"]:
                  spread_data["spread_history"].pop(0)

              spread_std = np.std(spread_data["spread_history"])

              zscore = (
                  spread - self.params[Product.SPREAD_INTER]["default_spread_mean"]
              ) / spread_std

            if zscore >= self.params[Product.SPREAD_INTER]["zscore_threshold"]:
                  if basket_position != -self.params[Product.SPREAD_B2]["target_position"]:
                      return self.##PUT IN EXECUTION HERE MICHAEL###(
                          -self.params[Product.SPREAD_B2]["target_position"],
                          basket_position,
                          order_depths,
                      )

            if zscore <= -self.params[Product.SPREAD_INTER]["zscore_threshold"]:
                  if basket_position != self.params[Product.SPREAD_INTER]["target_position"]:
                      return self.##PUT IN EXECUTION HERE MICHAEL###(
                          self.params[Product.SPREAD_INTER]["target_position"],
                          basket_position,
                          order_depths,
                      )

              spread_data["prev_zscore"] = zscore
              return None