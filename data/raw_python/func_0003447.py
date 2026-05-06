def updateorders(self):
        """ Update the orders
        """
        log.info("Replacing orders")

        # Canceling orders
        self.cancelall()

        # Target
        target = self.bot.get("target", {})
        price = self.getprice()

        # prices
        buy_price = price * (1 - target["offsets"]["buy"] / 100)
        sell_price = price * (1 + target["offsets"]["sell"] / 100)

        # Store price in storage for later use
        self["feed_price"] = float(price)

        # Buy Side
        if float(self.balance(self.market["base"])) < buy_price * target["amount"]["buy"]:
            InsufficientFundsError(Amount(target["amount"]["buy"] * float(buy_price), self.market["base"]))
            self["insufficient_buy"] = True
        else:
            self["insufficient_buy"] = False
            self.market.buy(
                buy_price,
                Amount(target["amount"]["buy"], self.market["quote"]),
                account=self.account
            )

        # Sell Side
        if float(self.balance(self.market["quote"])) < target["amount"]["sell"]:
            InsufficientFundsError(Amount(target["amount"]["sell"], self.market["quote"]))
            self["insufficient_sell"] = True
        else:
            self["insufficient_sell"] = False
            self.market.sell(
                sell_price,
                Amount(target["amount"]["sell"], self.market["quote"]),
                account=self.account
            )

        pprint(self.execute())