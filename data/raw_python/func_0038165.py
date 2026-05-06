def getaccountaddress(self, user_id=""):
        """Get the coin address associated with a user id.

        If the specified user id does not yet have an address for this
        coin, then generate one.

        Args:
          user_id (str): this user's unique identifier

        Returns:
          str: Base58Check address for this account

        """
        address = self.rpc.call("getaccountaddress", user_id)
        self.logger.debug("Your", self.coin, "address is", address)
        return address