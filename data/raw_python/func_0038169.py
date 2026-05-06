def sendfrom(self, user_id, dest_address, amount, minconf=1):
        """
        Send coins from user's account.

        Args:
          user_id (str): this user's unique identifier
          dest_address (str): address which is to receive coins
          amount (str or Decimal): amount to send (eight decimal points)
          minconf (int): ensure the account has a valid balance using this
                         many confirmations (default=1)

        Returns:
          str: transaction ID

        """
        amount = Decimal(amount).quantize(self.quantum, rounding=ROUND_HALF_EVEN)
        txhash = self.rpc.call("sendfrom",
            user_id, dest_address, float(str(amount)), minconf
        )
        self.logger.debug("Send %s %s from %s to %s" % (str(amount), self.coin,
                                                        str(user_id), dest_address))
        self.logger.debug("Transaction hash: %s" % txhash)
        return txhash