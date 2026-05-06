def payment(self, origin, destination, amount):
        """Convenience method for sending Bitcoins.

        Send coins from origin to destination. Calls record_tx to log the
        transaction to database.  Uses free, instant "move" transfers
        if addresses are both local (in the same wallet), and standard
        "sendfrom" transactions otherwise.

        The sender is required to be specified by user_id (account label);
        however, the recipient can be specified either by Bitcoin address
        (anyone) or user_id (if the user is local).

        Payment tries sending Bitcoins in this order:
          1. "move" from account to account (local)
          2. "move" from account to address (local)
          3. "sendfrom" account to address (broadcast)

        Args:
          origin (str): user_id of the sender
          destination (str): coin address or user_id of the recipient
          amount (str, Decimal, number): amount to send

        Returns:
          bool: True if successful, False otherwise

        """
        if type(amount) != Decimal:
            amount = Decimal(amount)
        if amount <= 0:
            raise Exception("Amount must be a positive number")

        # Check if the destination is within the same wallet;
        # if so, we can use the fast (and free) "move" command
        all_addresses = []
        accounts = self.listaccounts()
        if origin in accounts:
            if destination in accounts:
                with self.openwallet():
                    result = self.move(origin, destination, amount)
                return self.record_tx(origin, None, amount,
                                      result, destination)
            for account in accounts:
                addresses = self.getaddressesbyaccount(account)
                if destination in addresses:
                    with self.openwallet():
                        result = self.move(origin, account, amount)
                    return self.record_tx(origin, destination, amount,
                                          result, account)

            # Didn't find anything, so use "sendfrom" instead
            else:
                with self.openwallet():
                    txhash = self.sendfrom(origin, destination, amount)
                return self.record_tx(origin, destination, amount, txhash)