def register(self, service, name=''):
        """
        Exposes a given service to this API.
        """
        # expose a sub-factory
        if isinstance(service, ApiFactory):
            self.services[name] = (service.factory, None)

        # expose a module dynamically as a service
        elif inspect.ismodule(service):
            name = name or service.__name__.split('.')[-1]

            # exclude endpoints with patterns
            for obj in vars(service).values():
                endpoint = getattr(obj, 'endpoint', None)
                if isinstance(endpoint, Endpoint) and endpoint.pattern:
                    route = Route('', endpoint.pattern)
                    self.routes.append((route, endpoint))

            self.services[name] = (ModuleService, service)

        # expose a class dynamically as a service
        elif inspect.isclass(service):
            name = name or service.__name__
            self.services[name] = (ClassService, service)

        # expose an endpoint directly
        elif isinstance(getattr(service, 'endpoint', None), Endpoint):
            if service.endpoint.pattern:
                route = Route('', service.endpoint.pattern)
                self.routes.append((route, service.endpoint))
            else:
                self.services[service.endpoint.name] = (service.endpoint, None)

        # expose a scope
        elif isinstance(service, dict):
            for srv in service.values():
                try:
                    self.register(srv)
                except RuntimeError:
                    pass

        # expose a list of services
        elif isinstance(service, list):
            for srv in service:
                try:
                    self.register(srv)
                except RuntimeError:
                    pass

        # expose a service directly
        else:
            raise RuntimeError('Invalid service provide: {0} ({1}).'.format(service, type(service)))