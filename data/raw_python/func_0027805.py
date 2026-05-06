def get_wsgi_server(self, name=None, defaults=None):
        """
        Reads the configuration source and finds and loads a WSGI server
        defined by the server entry with the name ``name`` per the PasteDeploy
        configuration format and loading mechanism.

        :param name: The named WSGI server to find, load and return. Defaults
            to ``None`` which becomes ``main`` inside
            :func:`paste.deploy.loadserver`.
        :param defaults: The ``global_conf`` that will be used during server
            instantiation.
        :return: A WSGI server runner callable which accepts a WSGI app.

        """
        name = self._maybe_get_default_name(name)
        defaults = self._get_defaults(defaults)
        return loadserver(
            self.pastedeploy_spec,
            name=name,
            relative_to=self.relative_to,
            global_conf=defaults,
        )