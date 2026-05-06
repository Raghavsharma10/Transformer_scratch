def add_resource(self, handler, uri, methods=frozenset({'GET'}),
                     host=None, strict_slashes=None, version=None, name=None,
                     **kwargs):
        """
        Create a blueprint resource route from a function.

        :param uri: endpoint at which the route will be accessible.
        :param methods: list of acceptable HTTP methods.
        :param host:
        :param strict_slashes:
        :param version:
        :param name: user defined route name for url_for
        :return: function or class instance

        Accepts any keyword argument that will be passed to the app resource.
        """
        self.resource(uri=uri, methods=methods, host=host,
                      strict_slashes=strict_slashes, version=version,
                      name=name, **kwargs)(handler)