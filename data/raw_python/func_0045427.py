def connect(self, address, **kws):
        """Connect to a remote socket at _address_. """
        return yield_(Connect(self, address, timeout=self._timeout, **kws))