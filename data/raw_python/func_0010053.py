def get(self, key, default_value=__NoDefaultSpecified__):
        '''
        Gets the value from the yaml config based on the key.

        No type casting is performed, any type casting should be 
        performed by the caller.

        Args:
            key (str) - Config setting key.

        Kwargs:
            default_value - Default value to return if config is not specified.

        Returns:
            Returns value stored in config file.

        '''
        # First attempt to get the var from OS enviornment.
        os_env_string = ConfigReader.ENV_PREFIX + key
        os_env_string = os_env_string.replace(".", "_")
        if type(os.getenv(os_env_string)) != NoneType:
            return os.getenv(os_env_string)

        # Otherwise search through config files.
        for data_map in self._dataMaps:
            try:
                if "." in key:
                    # this is a multi levl string
                    namespaces = key.split(".")
                    temp_var = data_map
                    for name in namespaces:
                        temp_var = temp_var[name]
                    return temp_var
                else:
                    value = data_map[key]
                    return value
            except (AttributeError, TypeError, KeyError):
                pass

        if default_value == self.__NoDefaultSpecified__:
            raise KeyError(u("Key '{0}' does not exist").format(key))
        else:
            return default_value