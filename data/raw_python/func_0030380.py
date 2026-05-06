def stopReceivingBoxes(self, reason):
        """
        Stop all the L{IBoxReceiver}s which have been added to this router.
        """
        for routeName, route in self._routes.iteritems():
            route.stop(reason)
        self._routes = None