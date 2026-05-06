def _existing(self, email):
        """
        Return the existing L{EmailAddress} item with the given address, or
        C{None} if there isn't one.
        """
        return self.store.findUnique(
            EmailAddress,
            EmailAddress.address == email,
            default=None)