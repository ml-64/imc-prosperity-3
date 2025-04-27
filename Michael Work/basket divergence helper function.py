def basket_divergence(self, order_depths):

    # Linear regression on croissants to jams
    def croissant_prediction(x):
        return 1.69473601*x - 716.7409375462321

    # FIND SPREAD BETWEEN MODIFIED CROISSANTS AND JAMS PRICE
    croissant_price = self.mid_price(order_depths.get("CROISSANTS", {}))
    jams_price = self.mid_price(order_depths.get("JAMS",{}))

    mod_croissant_price = croissant_prediction(croissant_price)

    z_score = (jams_price - mod_croissant_price) / 54.85905218876869

    if abs(z_score) > 2:
        # WE ARE OVERPRICED ON JAMS, UNDERPRICED ON CROISSANTS; THEREFORE, SHORT JAMS AND BUY CROISSANTS #
        if z_score > 0: return -1
        else: return 1
    else: return 0