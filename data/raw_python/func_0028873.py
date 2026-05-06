def __set_route(self, type_route, route):
        """
        Sets the given type_route and route to the route mapping
        :rtype: object
        """
        if type_route in self.__routes:
            if not self.verify_route_already_bound(type_route, route):
                self.__routes[type_route].append(route)
        else:
            self.__routes[type_route] = [route]
        return RouteMapping