def get_wsgi_filter(self, name=None, defaults=None):
        """Reads the configuration soruce and finds and loads a WSGI filter
        defined by the filter entry with the name ``name`` per the PasteDeploy
        configuration format and loading mechanism.

        :param name: The named WSGI filter to find, load and return. Defaults
            to ``None`` which becomes ``main`` inside
            :func:`paste.deploy.loadfilter`.
        :param defaults: The ``global_conf`` that will be used during filter
            instantiation.
        :return: A callable that can filter a WSGI application.
        """
        name = self._maybe_get_default_name(name)
        defaults = self._get_defaults(defaults)
        return loadfilter(
            self.pastedeploy_spec,
            name=name,
            relative_to=self.relative_to,
            global_conf=defaults,
        )