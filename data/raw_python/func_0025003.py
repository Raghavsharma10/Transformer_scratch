def authenticate(self):
        """
        Authenticate into the UAA instance as the admin user.
        """
        # Make sure we've stored uri for use
        predix.config.set_env_value(self.use_class, 'uri', self._get_uri())

        self.uaac = predix.security.uaa.UserAccountAuthentication()
        self.uaac.authenticate('admin', self._get_admin_secret(),
                use_cache=False)
        self.is_admin = True