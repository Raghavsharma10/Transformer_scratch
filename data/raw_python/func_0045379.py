def connectionMade(self):
        """Keep a reference to the protocol on the factory, and uses the
        factory's store to find multiplexed connection factories.

        Unfortunately, we can't add the protocol by TLS certificate
        fingerprint, because the TLS handshake won't have completed
        yet, so ``self.transport.getPeerCertificate()`` is still
        ``None``.

        """
        self.factory.protocols.add(self)
        self._factories = multiplexing.FactoryDict(self.store)
        super(AMP, self).connectionMade()