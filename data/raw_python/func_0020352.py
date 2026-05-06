def init_app(self, app, session=None, parameters=None):
        """Initializes snow extension

        Set config default and find out which client type to use

        :param app: App passed from constructor or directly to init_app (factory)
        :param session: requests-compatible session to pass along to init_app
        :param parameters: `ParamsBuilder` object passed to `Client` after instantiation
        :raises:
            - ConfigError - if unable to determine client type
        """

        if parameters is not None and not isinstance(parameters, ParamsBuilder):
            raise InvalidUsage("parameters should be a pysnow.ParamsBuilder object, not %r" % type(parameters).__name__)

        self._session = session
        self._parameters = parameters

        app.config.setdefault('SNOW_INSTANCE', None)
        app.config.setdefault('SNOW_HOST', None)
        app.config.setdefault('SNOW_USER', None)
        app.config.setdefault('SNOW_PASSWORD', None)
        app.config.setdefault('SNOW_OAUTH_CLIENT_ID', None)
        app.config.setdefault('SNOW_OAUTH_CLIENT_SECRET', None)
        app.config.setdefault('SNOW_USE_SSL', True)

        if app.config['SNOW_OAUTH_CLIENT_ID'] and app.config['SNOW_OAUTH_CLIENT_SECRET']:
            self._client_type_oauth = True
        elif self._session or (app.config['SNOW_USER'] and app.config['SNOW_PASSWORD']):
            self._client_type_basic = True
        else:
            raise ConfigError("You must supply user credentials, a session or OAuth credentials to use flask-snow")