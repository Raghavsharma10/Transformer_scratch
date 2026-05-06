def get_config(self, configfile):
        """
            Loads the configuration from the JSON file, and returns it.
        :param configfile: Path to the configuration file
        """
        with open(configfile) as config_file:
            config = json.load(config_file)
        return config