def lookup(self, request_class: Request) -> Callable[[Request], BrightsideMessage]:
        """
        Looks up the message mapper function associated with this class. Function should take in a Request derived class
         and return a BrightsideMessage derived class, for sending on the wire
        :param request_class:
        :return:
        """
        key = request_class.__class__.__name__
        if key not in self._registry:
            raise ConfigurationException("There is no message mapper associated with this key; we require a mapper")
        else:
            return self._registry[key]