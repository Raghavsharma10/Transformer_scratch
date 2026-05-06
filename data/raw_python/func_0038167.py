def listtransactions(self, user_id="", count=10, start_at=0):
        """List all transactions associated with this account.

        Args:
          user_id (str): this user's unique identifier
          count (int): number of transactions to return (default=10)
          start_at (int): start the list at this transaction (default=0)

        Returns:
          list [dict]: transactions associated with this user's account

        """
        txlist = self.rpc.call("listtransactions", user_id, count, start_at)
        self.logger.debug("Got transaction list for " + str(user_id))
        return txlist