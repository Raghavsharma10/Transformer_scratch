def locateChild(self, ctx, segments):
        """
        Retrieve a L{SharingIndex} for a particular user, or rend.NotFound.
        """
        store = _storeFromUsername(
            self.loginSystem.store, segments[0].decode('utf-8'))
        if store is None:
            return rend.NotFound
        return (SharingIndex(store, self.webViewer), segments[1:])