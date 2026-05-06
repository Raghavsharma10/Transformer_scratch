def save_configuration_file(self):
        """
        Save all configuration into file
        Only if config file does not yet exist or configuration was changed
        """
        if os.path.exists(self.config_file) and not self.config_changed:
            return
        dirname = os.path.dirname(self.config_file)
        try:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        except (OSError, IOError) as e:
            self.logger.warning("Could not make directory for configuration file: {0}".
                                format(utils.exc_as_decoded_string(e)))
            return
        try:
            with open(self.config_file, 'w') as file:
                csvwriter = csv.writer(file, delimiter='=', escapechar='\\',
                                       lineterminator='\n', quoting=csv.QUOTE_NONE)
                for key, value in self.config_dict.items():
                    csvwriter.writerow([key, value])
            self.config_changed = False
        except (OSError, IOError) as e:
            self.logger.warning("Could not save configuration file: {0}".\
                format(utils.exc_as_decoded_string(e)))