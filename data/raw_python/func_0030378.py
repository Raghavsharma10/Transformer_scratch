def startReceivingBoxes(self, sender):
        """
        Initialize route tracking objects.
        """
        self._sender = sender
        for routeName, route in self._unstarted.iteritems():
            # Any route which has been bound but which does not yet have a
            # remote route name should not yet be started.  These will be
            # started in Route.connectTo.
            if route.remoteRouteName is not _unspecified:
                route.start()
        self._routes = self._unstarted
        self._unstarted = None