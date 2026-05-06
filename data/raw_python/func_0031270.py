def init_config(self, app):
        """Initialize configuration."""
        for k in dir(config):
            if k.startswith('JSONSCHEMAS_'):
                app.config.setdefault(k, getattr(config, k))

        host_setting = app.config['JSONSCHEMAS_HOST']
        if not host_setting or host_setting == 'localhost':
            app.logger.warning('JSONSCHEMAS_HOST is set to {0}'.format(
                host_setting))