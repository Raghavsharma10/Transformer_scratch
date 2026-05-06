def put(self, route: str(), callback: object()):
        """
        Binds a PUT route with the given callback
        :rtype: object
        """
        self.__set_route('put', {route: callback})
        return RouteMapping