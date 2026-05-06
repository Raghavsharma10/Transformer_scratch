def get(self, route: str(), callback: object()):
        """
        Binds a GET route with the given callback 
        :rtype: object
        """
        self.__set_route('get', {route: callback})
        return RouteMapping