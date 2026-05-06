def register_route(self, name, route):
        """Register a route handler

        :param name: Name of the route
        :param route: Route handler
        """
        try:
            self.routes[name] = route.handle
        except Exception as e:
            print('could not import handle, maybe something wrong ',
                  'with your code?')
            print(e)