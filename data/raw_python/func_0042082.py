def find_config_file(self):
        """
            Find where our config file is if there is any

            If the value for the config file is a default and it doesn't exist
            then it is silently ignored.

            If however, the value isn't a default and it doesn't exist, an error is raised
        """
        filename = self.values.get('config_file', Default('noy.json'))

        ignore_missing = False
        if isinstance(filename, Default):
            filename = filename.val
            ignore_missing = True

        filename = os.path.abspath(filename)
        if os.path.exists(filename):
            return filename
        elif not ignore_missing:
            raise MissingConfigFile("Config file doesn't exist at {}".format(filename))