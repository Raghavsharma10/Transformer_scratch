def resource(self, uri, methods=frozenset({'GET'}), **kwargs):
        """
        Decorates a function to be registered as a resource route.

        :param uri: path of the URL
        :param methods: list or tuple of methods allowed
        :param host:
        :param strict_slashes:
        :param stream:
        :param version:
        :param name: user defined route name for url_for
        :param filters: List of callable that will filter request and
                        response data
        :param validators: List of callable added to the filter list.

        :return: A decorated function
        """
        def decorator(f):
            if kwargs.get('stream'):
                f.is_stream = kwargs['stream']
            self.add_resource(f, uri=uri, methods=methods, **kwargs)

        return decorator