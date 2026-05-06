def setup(self, options=None, extractor=None):
        """
            Put options onto the config and put anything from a config file onto the config.

            If extractor is specified, it is used to extract values from the options dictionary
        """
        # Get our programmatic options
        self._util.use_options(options, extractor)

        # Overwrite non defaults in self.values with values from config
        self._util.use_config_file()