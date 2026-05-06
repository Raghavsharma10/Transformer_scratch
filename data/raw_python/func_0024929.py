def get_service_env_value(self, key):
        """
        Get a env variable as defined by the service admin
        :param key: the base of the key to use
        :return: the env if it exists
        """
        service_key = predix.config.get_env_key(self, key)
        value = os.environ[service_key]
        if not value:
            raise ValueError("%s env unset" % key)
        return value