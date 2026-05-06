def resource(self, uri, methods=frozenset({'GET'}), host=None,
                 strict_slashes=None, stream=False, version=None, name=None,
                 **kwargs):
        """
        Create a blueprint resource route from a decorated function.

        :param uri: endpoint at which the route will be accessible.
        :param methods: list of acceptable HTTP methods.
        :param host:
        :param strict_slashes:
        :param version:
        :param name: user defined route name for url_for
        :return: function or class instance

        Accepts any keyword argument that will be passed to the app resource.
        """
        if strict_slashes is None:
            strict_slashes = self.strict_slashes

        def decorator(handler):
            self.resources.append((
                FutureRoute(handler, uri, methods, host, strict_slashes,
                            stream, version, name),
                kwargs))

            return handler
        return decorator