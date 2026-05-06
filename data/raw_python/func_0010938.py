def get(cls):
        """Get the current API key.
        if one has not been given via 'set' the env var STEAMODD_API_KEY will
        be checked instead.
        """
        apikey = cls.__api_key or cls.__api_key_env_var

        if apikey:
            return apikey
        else:
            raise APIKeyMissingError("API key not set")