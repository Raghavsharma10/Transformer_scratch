def save_settings(self, path, settings, readable=False):
        """
        Save settings to file

        :param path: File path to save
        :type path: str | unicode
        :param settings: Settings to save
        :type settings: dict
        :param readable: Format file to be human readable (default: False)
        :type readable: bool
        :rtype: None
        :raises IOError: If empty path or error writing file
        :raises TypeError: Settings is not a dict
        """
        if not isinstance(settings, dict):
            raise TypeError("Expected settings to be dict")
        return self.save_file(path, settings, readable)