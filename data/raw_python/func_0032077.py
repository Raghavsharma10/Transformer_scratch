def getFactory(self):
        """
        Return a server factory which creates AMP protocol instances.
        """
        factory = ServerFactory()
        def protocol():
            proto = CredReceiver()
            proto.portal = Portal(
                self.loginSystem,
                [self.loginSystem,
                 OneTimePadChecker(self._oneTimePads)])
            return proto
        factory.protocol = protocol
        return factory