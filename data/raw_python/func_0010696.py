def _update_from_file(self, filename):
        """ Helper method to update an existing configuration with the values from a file.

        Loads a configuration file and replaces all values in the existing configuration
        dictionary with the values from the file.

        Args:
            filename (str): The path and name to the configuration file.
        """
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as config_file:
                    yaml_dict = yaml.safe_load(config_file.read())
                    if yaml_dict is not None:
                        self._update_dict(self._config, yaml_dict)
            except IsADirectoryError:
                raise ConfigLoadError(
                    'The specified configuration file is a directory not a file')
        else:
            raise ConfigLoadError('The config file {} does not exist'.format(filename))