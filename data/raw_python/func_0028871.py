def head(self, route: str(), callback: object()):
        """
        Binds a HEAD route with the given callback
        :rtype: object
        """
        self.__set_route('head', {route: callback})
        return RouteMapping