def cancelall(self):
        """ Cancel all orders of this bot
        """
        if self.orders:
            return self.bitshares.cancel(
                [o["id"] for o in self.orders],
                account=self.account
            )