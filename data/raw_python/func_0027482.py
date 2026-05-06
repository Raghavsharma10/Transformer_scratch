def register(self, request_class: Request, handler_factory: Callable[[], Handler]) -> None:
        """
        Register the handler for the command
        :param request_class: The command or event to dispatch. It must implement getKey()
        :param handler_factory: A factory method to create the handler to dispatch to
        :return:
        """
        key = request_class.__name__
        is_command = request_class.is_command()
        is_event = request_class.is_event()
        is_present = key in self._registry
        if is_command and is_present:
            raise ConfigurationException("A handler for this request has already been registered")
        elif is_event and is_present:
            self._registry[key].append(handler_factory)
        elif is_command or is_event:
            self._registry[key] = [handler_factory]