def delete(self, route: str(), callback: object()):
        """
        Binds a PUT route with the given callback
        :rtype: object
        """
        self.__set_route('delete', {route: callback})
        return RouteMapping