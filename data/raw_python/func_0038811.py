def get_config(self, key_name):
        """
        Return configuration value

        Args:
            key_name (str): configuration key

        Returns:
            The value for the specified configuration key, or if not found
            in the config the default value specified in the Configuration Handler
            class specified inside this component

        """
        if key_name in self.config:
            return self.config.get(key_name)
        return self.Configuration.default(key_name, inst=self)