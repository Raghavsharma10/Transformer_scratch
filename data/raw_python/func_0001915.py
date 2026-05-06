def load_settings(self, path):
        """
        Load settings dict

        :param path: Path to settings file
        :type path: str | unicode
        :return: Loaded settings
        :rtype: dict
        :raises IOError: If file not found or error accessing file
        :raises TypeError: Settings file does not contain dict
        """
        res = self.load_file(path)
        if not isinstance(res, dict):
            raise TypeError("Expected settings to be dict")
        return res