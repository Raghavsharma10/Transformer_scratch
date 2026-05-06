def getprice(self):
        """ Here we obtain the price for the quote and make sure it has
            a feed price
        """
        target = self.bot.get("target", {})
        if target.get("reference") == "feed":
            assert self.market == self.market.core_quote_market(), "Wrong market for 'feed' reference!"
            ticker = self.market.ticker()
            price = ticker.get("quoteSettlement_price")
            assert abs(price["price"]) != float("inf"), "Check price feed of asset! (%s)" % str(price)
        return price