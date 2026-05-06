def getbalance(self, user_id="", as_decimal=True):
        """Calculate the total balance in all addresses belonging to this user.

        Args:
          user_id (str): this user's unique identifier
          as_decimal (bool): balance is returned as a Decimal if True (default)
                             or a string if False

        Returns:
          str or Decimal: this account's total coin balance

        """
        balance = unicode(self.rpc.call("getbalance", user_id))
        self.logger.debug("\"" + user_id + "\"", self.coin, "balance:", balance)
        if as_decimal:
            return Decimal(balance)
        else:
            return balance