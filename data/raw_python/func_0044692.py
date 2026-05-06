def check_filepath(self, path, filename):
        """
        Check and return the final filepath to settings

        Args:
            path (str): Directory path where to search for settings file.
            filename (str): Filename to use to search for settings file.

        Raises:
            boussole.exceptions.SettingsBackendError: If determined filepath
                does not exists or is a directory.

        Returns:
            string: Settings file path, joining given path and filename.

        """
        settings_path = os.path.join(path, filename)

        if not os.path.exists(settings_path) or \
           not os.path.isfile(settings_path):
            msg = "Unable to find settings file: {}"
            raise SettingsBackendError(msg.format(settings_path))

        return settings_path