def set(self, **kwargs):
        '''
        Override existing settings, taking precedence over both user settings
        object and default settings. Useful for specific runtime requirements,
        such as overriding PORT or HOST.
        '''
        for lower_key, value in kwargs.items():
            if lower_key.lower() != lower_key:
                raise ValueError('Requires lowercase: %s' % lower_key)
            key = lower_key.upper()
            try:
                getattr(self, key)
            except (AttributeError, ConfigurationError):
                raise AttributeError('Cannot override %s' % key)
            self.overridden_settings[key] = value