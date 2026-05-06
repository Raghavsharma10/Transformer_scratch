def patch(self, route: str(), callback: object()):
        """
        Binds a PATCH route with the given callback
        :rtype: object
        """
        self.__set_route('patch', {route: callback})
        return RouteMapping