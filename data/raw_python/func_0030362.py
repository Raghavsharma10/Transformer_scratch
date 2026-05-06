def configure(self, config=None, **kwargs):
        """
        We expect all of our config (apart from the ENGINE) to be
        in a dictionary called 'config' in our INSTALLED_BACKENDS entry
        """
        self.config = config or {}
        for key in ['messaging_token', 'number']:
            if key not in self.config:
                msg = "Tropo backend config must set '%s'; config is %r" %\
                      (key, config)
                raise ImproperlyConfigured(msg)
        if kwargs:
            msg = "All tropo backend config should be within the `config`"\
                "entry of the backend dictionary"
            raise ImproperlyConfigured(msg)