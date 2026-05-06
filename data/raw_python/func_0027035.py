def _read_config(cls):
        """ Reads the configuration file if any
        """

        cls._config_parser = configparser.ConfigParser()
        cls._config_parser.read(cls._default_attribute_values_configuration_file_path)