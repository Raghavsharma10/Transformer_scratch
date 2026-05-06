def get_client_id(self):
        """
        Return the client id that should have all the
        needed scopes and authorities for the services
        in this manifest.
        """
        self._client_id = predix.config.get_env_value(predix.app.Manifest, 'client_id')
        return self._client_id