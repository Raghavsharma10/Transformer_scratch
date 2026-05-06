def connectionLost(self, reason):
        """Lose the reference to the protocol on the factory.

        """
        self.factory.protocols.remove(self)
        super(AMP, self).connectionLost(reason)