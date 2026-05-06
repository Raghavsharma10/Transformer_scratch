def navigate(self, name, *args):
        """Navigate to a route
        * 
        * @param  {String} name Route name
        * @param  {*}      arg  A single argument to pass to the route handler
        */
        """
        if name not in self.routes:
            raise Exception('invalid route name \'%s\'' % name)
        elif callable(self.routes[name]):
            return self.routes[name](self, *args)
        raise Exception('route %s not callable', name)