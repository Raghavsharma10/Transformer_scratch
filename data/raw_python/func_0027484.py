def register(self, request_class: Request, mapper_func: Callable[[Request], BrightsideMessage]) -> None:
        """Adds a message mapper to a factory, using the requests key
        :param mapper_func: A callback that creates a BrightsideMessage from a Request
        :param request_class: A request type
        """

        key = request_class.__name__
        if key not in self._registry:
            self._registry[key] = mapper_func
        else:
            raise ConfigurationException("There is already a message mapper defined for this key; there can be only one")