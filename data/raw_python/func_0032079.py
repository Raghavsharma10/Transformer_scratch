def indirect(self, interface):
        """
        Create a L{Router} to handle AMP boxes received over an AMP connection.
        """
        if interface is IBoxReceiver:
            router = Router()
            connector = self.connectorFactory(router)
            router.bindRoute(connector, None).connectTo(None)
            return router
        raise NotImplementedError()