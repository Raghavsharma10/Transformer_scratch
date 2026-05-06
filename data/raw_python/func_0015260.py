def load_configuration_file(self):
        """
        Load all configuration from file
        """
        if not os.path.exists(self.config_file):
            return
        try:
            with open(self.config_file, 'r') as file:
                csvreader = csv.reader(file, delimiter='=',
                                       escapechar='\\', quoting=csv.QUOTE_NONE)
                for line in csvreader:
                    if len(line) == 2:
                        key, value = line
                        self.config_dict[key] = value
                    else:
                        self.config_dict = dict()
                        self.logger.warning("Malformed configuration file {0}, ignoring it.".
                                            format(self.config_file))
                        return
        except (OSError, IOError) as e:
            self.logger.warning("Could not load configuration file: {0}".\
                format(utils.exc_as_decoded_string(e)))