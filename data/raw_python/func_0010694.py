def load_from_file(self, filename=None, *, strict=True):
        """ Load the configuration from a file.

        The location of the configuration file can either be specified directly in the
        parameter filename or is searched for in the following order:

            1. In the environment variable given by LIGHTFLOW_CONFIG_ENV
            2. In the current execution directory
            3. In the user's home directory

        Args:
            filename (str): The location and name of the configuration file.
            strict (bool): If true raises a ConfigLoadError when the configuration
                cannot be found.

        Raises:
            ConfigLoadError: If the configuration cannot be found.
        """
        self.set_to_default()

        if filename:
            self._update_from_file(filename)
        else:
            if LIGHTFLOW_CONFIG_ENV not in os.environ:
                if os.path.isfile(os.path.join(os.getcwd(), LIGHTFLOW_CONFIG_NAME)):
                    self._update_from_file(
                        os.path.join(os.getcwd(), LIGHTFLOW_CONFIG_NAME))
                elif os.path.isfile(expand_env_var('~/{}'.format(LIGHTFLOW_CONFIG_NAME))):
                    self._update_from_file(
                        expand_env_var('~/{}'.format(LIGHTFLOW_CONFIG_NAME)))
                else:
                    if strict:
                        raise ConfigLoadError('Could not find the configuration file.')
            else:
                self._update_from_file(expand_env_var(os.environ[LIGHTFLOW_CONFIG_ENV]))

        self._update_python_paths()