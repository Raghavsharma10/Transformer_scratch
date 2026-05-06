def post(self, route: str(), callback: object()):
        """
        Binds a POST route with the given callback
        :rtype: object
        """
        self.__set_route('post', {route: callback})
        return RouteMapping