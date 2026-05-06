def get_wsgi_app(self, name=None, defaults=None):
        """
        Reads the configuration source and finds and loads a WSGI
        application defined by the entry with name ``name`` per the
        PasteDeploy configuration format and loading mechanism.

        :param name: The named WSGI app to find, load and return. Defaults to
            ``None`` which becomes ``main`` inside
            :func:`paste.deploy.loadapp`.
        :param defaults: The ``global_conf`` that will be used during app
            instantiation.
        :return: A WSGI application.

        """
        name = self._maybe_get_default_name(name)
        defaults = self._get_defaults(defaults)
        return loadapp(
            self.pastedeploy_spec,
            name=name,
            relative_to=self.relative_to,
            global_conf=defaults,
        )