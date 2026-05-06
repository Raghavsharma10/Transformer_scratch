def connectionLost(self, reason):
        """
        If a login has happened, perform a logout.
        """
        AMP.connectionLost(self, reason)
        if self.logout is not None:
            self.logout()
            self.boxReceiver = self.logout = None