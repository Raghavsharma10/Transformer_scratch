def connectTo(self, remoteRouteName):
        """
        Set the name of the route which will be added to outgoing boxes.
        """
        self.remoteRouteName = remoteRouteName
        # This route must not be started before its router is started.  If
        # sender is None, then the router is not started.  When the router is
        # started, it will start this route.
        if self.router._sender is not None:
            self.start()