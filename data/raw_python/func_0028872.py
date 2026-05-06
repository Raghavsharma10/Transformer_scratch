def options(self, route: str(), callback: object()):
        """
        Binds a OPTIONS route with the given callback
        :rtype: object
        """
        self.__set_route('options', {route: callback})
        return RouteMapping