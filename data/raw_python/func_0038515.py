def route(self, resource):
        """
        route
        """
        route = self.routes.get(resource, Route(resource))
        self.routes.update({resource: route})
        return route