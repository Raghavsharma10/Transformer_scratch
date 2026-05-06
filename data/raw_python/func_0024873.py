def get_client_secret(self):
        """
        Return the client secret that should correspond with
        the client id.
        """
        self._client_secret = predix.config.get_env_value(predix.app.Manifest, 'client_secret')
        return self._client_secret