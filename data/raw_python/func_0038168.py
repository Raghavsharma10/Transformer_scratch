def move(self, fromaccount, toaccount, amount, minconf=1):
        """Send coins between accounts in the same wallet.

        If the receiving account does not exist, it is automatically
        created (but not automatically assigned an address).

        Args:
          fromaccount (str): origin account
          toaccount (str): destination account
          amount (str or Decimal): amount to send (8 decimal points)
          minconf (int): ensure the account has a valid balance using this
                         many confirmations (default=1) 

        Returns:
          bool: True if the coins are moved successfully, False otherwise

        """
        amount = Decimal(amount).quantize(self.quantum, rounding=ROUND_HALF_EVEN)
        return self.rpc.call("move",
            fromaccount, toaccount, float(str(amount)), minconf
        )