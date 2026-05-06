def _validate_configuration(self):
        """
        Validates that required parameters are present.
        """

        if not self.access_token:
            raise ConfigurationException(
                'You will need to initialize a client with an Access Token'
            )
        if not self.api_url:
            raise ConfigurationException(
                'The client configuration needs to contain an API URL'
            )
        if not self.default_locale:
            raise ConfigurationException(
                'The client configuration needs to contain a Default Locale'
            )
        if not self.api_version or self.api_version < 1:
            raise ConfigurationException(
                'The API Version must be a positive number'
            )