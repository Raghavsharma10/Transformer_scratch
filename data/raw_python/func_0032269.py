def get_service(self, name):
        """
        Locates a remote service by name. The name can be a glob-like pattern
        (``"project.worker.*"``). If multiple services match the given name, a
        random instance will be chosen. There might be multiple services that
        match a given name if there are multiple services with the same name
        running, or when the pattern matches multiple different services.

        .. todo::

            Make this use self.io_loop to resolve the request. The current
            implementation is blocking and slow

        :param name: a pattern for the searched service.
        :return: a :py:class:`gemstone.RemoteService` instance
        :raises ValueError: when the service can not be located
        :raises ServiceConfigurationError: when there is no configured discovery strategy
        """
        if not self.discovery_strategies:
            raise ServiceConfigurationError("No service registry available")

        cached = self.remote_service_cache.get_entry(name)
        if cached:
            return cached.remote_service

        for strategy in self.discovery_strategies:
            endpoints = strategy.locate(name)
            if not endpoints:
                continue
            random.shuffle(endpoints)
            for url in endpoints:
                try:
                    service = get_remote_service_instance_for_url(url)
                    self.remote_service_cache.add_entry(name, service)
                    return service
                except ConnectionError:
                    continue  # could not establish connection, try next

        raise ValueError("Service could not be located")