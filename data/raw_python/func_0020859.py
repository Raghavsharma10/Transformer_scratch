def init_config(self, app):
        """Initialize configuration.

        :param app: An instance of :class:`flask.Flask`.
        """
        app.config.setdefault(
            'OAISERVER_BASE_TEMPLATE',
            app.config.get('BASE_TEMPLATE',
                           'invenio_oaiserver/base.html'))

        app.config.setdefault(
            'OAISERVER_REPOSITORY_NAME',
            app.config.get('THEME_SITENAME',
                           'Invenio-OAIServer'))

        # warn user if ID_PREFIX is not set
        if 'OAISERVER_ID_PREFIX' not in app.config:
            import socket
            import warnings

            app.config.setdefault(
                'OAISERVER_ID_PREFIX',
                'oai:{0}:recid/'.format(socket.gethostname()))
            warnings.warn(
                """Please specify the OAISERVER_ID_PREFIX configuration."""
                """default value is: {0}""".format(
                    app.config.get('OAISERVER_ID_PREFIX')))

        for k in dir(config):
            if k.startswith('OAISERVER_'):
                app.config.setdefault(k, getattr(config, k))