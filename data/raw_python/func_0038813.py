def bind(cls, app, *paths, methods=None, name=None, **kwargs):
        """Bind to the application.

        Generate URL, name if it's not provided.
        """
        paths = paths or ['/%s(/{%s})?/?' % (cls.name, cls.name)]
        name = name or "api.%s" % cls.name
        return super(RESTHandler, cls).bind(app, *paths, methods=methods, name=name, **kwargs)