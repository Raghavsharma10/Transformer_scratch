def init_app(self, app):
        """
        :param app: :class:`sanic.Sanic` instance to rate limit.
        """
        self.enabled = app.config.setdefault(C.ENABLED, True)
        self._swallow_errors = app.config.setdefault(
            C.SWALLOW_ERRORS, self._swallow_errors
        )
        self._storage_options.update(
            app.config.get(C.STORAGE_OPTIONS, {})
        )
        self._storage = storage_from_string(
            self._storage_uri
            or app.config.setdefault(C.STORAGE_URL, 'memory://'),
            **self._storage_options
        )
        strategy = (
            self._strategy
            or app.config.setdefault(C.STRATEGY, 'fixed-window')
        )
        if strategy not in STRATEGIES:
            raise ConfigurationError("Invalid rate limiting strategy %s" % strategy)
        self._limiter = STRATEGIES[strategy](self._storage)

        conf_limits = app.config.get(C.GLOBAL_LIMITS, None)
        if not self._global_limits and conf_limits:
            self._global_limits = [
                ExtLimit(
                    limit, self._key_func, None, False, None, None, None
                ) for limit in parse_many(conf_limits)
                ]
        app.request_middleware.append(self.__check_request_limit)