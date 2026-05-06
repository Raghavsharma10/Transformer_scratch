def lookup(self, request: Request) -> List[Callable[[], Handler]]:
        """
        Looks up the handler associated with a request - matches the key on the request to a registered handler
        :param request: The request we want to find a handler for
        :return:
        """
        key = request.__class__.__name__
        if key not in self._registry:
            if request.is_command():
                raise ConfigurationException("There is no handler registered for this request")
            elif request.is_event():
                return []  # type: Callable[[] Handler]

        return self._registry[key]