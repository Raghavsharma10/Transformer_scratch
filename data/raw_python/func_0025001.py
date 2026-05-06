def create(self, secret, **kwargs):
        """
        Create a new instance of the UAA service.  Requires a
        secret password for the 'admin' user account.
        """
        parameters = {"adminClientSecret": secret}
        self.service.create(parameters=parameters)

        # Store URI into environment variable
        predix.config.set_env_value(self.use_class, 'uri', self._get_uri())

        # Once we create it login
        self.authenticate()