def create_route_func(self, method):
        """
        create_route_func
        """
        def _route(resource, handler, schema=None):
            """
            _route
            """
            route = self.routes.get(resource, Route(resource))
            route.__getattribute__(method)(handler, schema)
            self.routes.update({resource: route})
            return self

        return _route