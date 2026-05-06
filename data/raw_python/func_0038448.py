def callRemote(self, *a, **kw):
        """
        Make a callRemote request of the JSONRPCClientFactory.
        """
        if not self.running:
            return defer.fail(ServiceStopped())
        return self.clientFactory.callRemote(*a, **kw)