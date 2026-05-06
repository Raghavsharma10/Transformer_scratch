def _validate_config(self):
        """
        Validate configuration file.
        :raises: RuntimeError
        """
        # while set().issubset() is easier, we want to tell the user the names
        # of any invalid keys
        bad_keys = []
        for k in self._config.keys():
            if k not in self._example.keys():
                bad_keys.append(k)
        if len(bad_keys) > 0:
            raise InvalidConfigError('Invalid keys: %s' % bad_keys)
        # endpoints
        if 'endpoints' not in self._config or len(
                self._config['endpoints']) < 1:
            raise InvalidConfigError('configuration must have '
                                     'at least one endpoint')
        for ep in self._config['endpoints']:
            if sorted(
                    self._config['endpoints'][ep].keys()
            ) != ['method', 'queues']:
                raise InvalidConfigError('Endpoint %s configuration keys must '
                                         'be "method" and "queues".' % ep)
            meth = self._config['endpoints'][ep]['method']
            if meth not in self._allowed_methods:
                raise InvalidConfigError('Endpoint %s method %s not allowed '
                                         '(allowed methods: %s'
                                         ')' % (ep, meth,
                                                self._allowed_methods))
        levels = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET']
        if ('logging_level' in self._config and
                self._config['logging_level'] not in levels):
            raise InvalidConfigError('logging_level must be one of %s' % levels)
        """
        'api_gateway_method_settings': {
            'throttlingBurstLimit': None,
            'throttlingRateLimit': None
        },
        """
        if 'api_gateway_method_settings' not in self._config:
            return
        ms = self._config['api_gateway_method_settings']
        bad_keys = []
        for k in ms.keys():
            if k not in self._example['api_gateway_method_settings'].keys():
                bad_keys.append(k)
        if len(bad_keys) > 0:
            raise InvalidConfigError(
                'Invalid keys in "api_gateway_method_settings": %s' % bad_keys)
        if 'metricsEnabled' in ms and ms['metricsEnabled'] not in [True, False]:
            raise InvalidConfigError(
                'api_gateway_method_settings metricsEnabled key must be omitted'
                ' or a boolean')
        if ('loggingLevel' in ms and
                ms['loggingLevel'] not in ['OFF', 'INFO', 'ERROR']):
            raise InvalidConfigError(
                'api_gateway_method_settings loggingLevel must be omitted or '
                'one of "OFF", "INFO" or "ERROR"'
            )
        if ('metricsEnabled' in ms and
                ms['dataTraceEnabled'] not in [True, False]):
            raise InvalidConfigError(
                'api_gateway_method_settings dataTraceEnabled key must be '
                'omitted or a boolean')
        if ('throttlingBurstLimit' in ms and
                ms['throttlingBurstLimit'] is not None):
            try:
                assert ms['throttlingBurstLimit'] == int(
                    ms['throttlingBurstLimit'])
            except (AssertionError, ValueError, TypeError):
                raise InvalidConfigError(
                    'api_gateway_method_settings throttlingBurstLimit key must '
                    'be omitted, null or an integer'
                )
        if ('throttlingRateLimit' in ms and
                ms['throttlingRateLimit'] is not None):
            try:
                assert ms['throttlingRateLimit'] == float(
                    ms['throttlingRateLimit'])
            except (AssertionError, ValueError, TypeError):
                raise InvalidConfigError(
                    'api_gateway_method_settings throttlingRateLimit key must '
                    'be omitted, null or a Number (float/double)'
                )