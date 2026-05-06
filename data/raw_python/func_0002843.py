def add_resource(self, handler, uri, methods=frozenset({'GET'}),
                     **kwargs):
        """
        Register a resource route.

        :param handler: function or class instance
        :param uri: path of the URL
        :param methods: list or tuple of methods allowed
        :param host:
        :param strict_slashes:
        :param version:
        :param name: user defined route name for url_for
        :param filters: List of callable that will filter request and
                        response data
        :param validators: List of callable added to the filter list.

        :return: function or class instance
        """

        sanic_args = ('host', 'strict_slashes', 'version', 'name')
        view_kwargs = dict((k, v) for k, v in kwargs.items()
                           if k in sanic_args)

        filters = kwargs.get('filters', self.default_filters)
        validators = kwargs.get('validators', [])

        filter_list = list(filters) + list(validators)
        filter_options = {
            'filter_list': filter_list,
            'handler': handler,
            'uri': uri,
            'methods': methods
        }
        filter_options.update(kwargs)

        handler = self.init_filters(filter_list, filter_options)(handler)
        return self.add_route(handler=handler, uri=uri, methods=methods,
                              **view_kwargs)