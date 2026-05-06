def send(self, request: Request) -> None:
        """
        Dispatches a request. Expects one and one only target handler
        :param request: The request to dispatch
        :return: None, will throw a ConfigurationException if more than one handler factor is registered for the command
        """

        handler_factories = self._registry.lookup(request)
        if len(handler_factories) != 1:
            raise ConfigurationException("There is no handler registered for this request")
        handler = handler_factories[0]()
        handler.handle(request)