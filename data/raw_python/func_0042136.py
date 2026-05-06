def as_string(cls, **initkwargs):
        """
        Similar to the as_view classmethod except this method will
        render this view as a string. When rendering a view this way
        the request will always be routed to the get method.
        The default render_type is 'string' unless you specify
        something else. If you provide your own render_type be sure
        to specify a render class that returns a string.
        """

        if not 'render_type' in initkwargs:
            initkwargs['render_type'] = 'string'

        for key in initkwargs:
            if key in cls.http_method_names:
                raise TypeError(u"You tried to pass in the %s method name as a"
                                u" keyword argument to %s(). Don't do that."
                                % (key, cls.__name__))
            if not hasattr(cls, key):
                raise TypeError(u"%s() received an invalid keyword %r" % (
                    cls.__name__, key))

        def view(request, *args, **kwargs):
            try:
                self = cls(**initkwargs)
                self.request = request
                self.args = args
                self.kwargs = kwargs
                return self.get_as_string(request, *args, **kwargs)
            except http.Http404:
                return ""

        # take name and docstring from class
        update_wrapper(view, cls, updated=())

        return view