def register(self, *paths, methods=None, name=None):
        """Register handler to the API."""
        if isinstance(methods, str):
            methods = [methods]

        def wrapper(handler):

            if isinstance(handler, (FunctionType, MethodType)):
                handler = RESTHandler.from_view(handler, *(methods or ['GET']))

            if handler.name in self.handlers:
                raise muffin.MuffinException('Handler is already registered: %s' % handler.name)

            self.handlers[tuple(paths or ["/{0}/{{{0}}}".format(handler.name)])] = handler

            handler.bind(self.app, *paths, methods=methods, name=name or handler.name)
            return handler

        # Support for @app.register(func)
        if len(paths) == 1 and callable(paths[0]):
            view = paths[0]
            paths = []
            return wrapper(view)

        return wrapper