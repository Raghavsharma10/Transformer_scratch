def factory(self, request, parent=None, name=None):
        """
        Returns a new service for the given request.

        :param      request | <pyramid.request.Request>

        :return     <pyramid_restful.services.AbstractService>
        """
        traverse = request.matchdict['traverse']

        # show documentation at the root path
        if not traverse:
            return {}
        else:
            service = {}
            name = name or traverse[0]

            # look for direct pattern matches
            traversed = '/' + '/'.join(traverse)
            service_type = None
            service_object = None

            for route, endpoint in self.routes:
                result = route.match(traversed)
                if result is not None:
                    request.matchdict = result
                    request.endpoint = endpoint
                    break
            else:
                try:
                    service_type, service_object = self.services[name]
                except KeyError:
                    raise HTTPNotFound()

            if service_type:
                if isinstance(service_type, Endpoint):
                    service[name] = service_type
                elif service_object is None:
                    service[name] = service_type(request)
                else:
                    service[name] = service_type(request, service_object)

            request.api_service = service
            return service