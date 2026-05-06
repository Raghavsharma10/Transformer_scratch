def get_wsgi_app_settings(self, name=None, defaults=None):
        """
        Return an :class:`collections.OrderedDict` representing the
        application config for a WSGI application named ``name`` in the
        PasteDeploy config file specified by ``self.uri``.

        ``defaults``, if passed, should be a dictionary used as variable
        assignments like ``{'http_port': 8080}``.  This is useful if e.g.
        ``%(http_port)s`` is used in the config file.

        If the ``name`` is None, this will attempt to parse the name from
        the ``config_uri`` string expecting the format ``inifile#name``.
        If no name is found, the name will default to "main".

        :param name: The named WSGI app for which to find the settings.
            Defaults to ``None`` which becomes ``main``
            inside :func:`paste.deploy.loadapp`.
        :param defaults: The ``global_conf`` that will be used during settings
            generation.
        :return: A :class:`plaster_pastedeploy.ConfigDict` of key/value pairs.

        """
        name = self._maybe_get_default_name(name)
        defaults = self._get_defaults(defaults)
        conf = appconfig(
            self.pastedeploy_spec,
            name=name,
            relative_to=self.relative_to,
            global_conf=defaults,
        )
        return ConfigDict(conf.local_conf, conf.global_conf, self)