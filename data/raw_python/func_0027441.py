def publish(self, request: Request) -> None:
        """
        Dispatches a request. Expects zero or more target handlers
        :param request: The request to dispatch
        :return: None.
        """
        handler_factories = self._registry.lookup(request)
        for factory in handler_factories:
            handler = factory()
            handler.handle(request)