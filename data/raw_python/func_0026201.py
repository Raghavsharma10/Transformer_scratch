def reload(self):
        """
        Reload a ConfigObj from file.

        This method raises a ``ReloadError`` if the ConfigObj doesn't have
        a filename attribute pointing to a file.
        """
        if not isinstance(self.filename, string_types):
            raise ReloadError()

        filename = self.filename
        current_options = {}
        for entry in OPTION_DEFAULTS:
            if entry == 'configspec':
                continue
            current_options[entry] = getattr(self, entry)

        configspec = self._original_configspec
        current_options['configspec'] = configspec

        self.clear()
        self._initialise(current_options)
        self._load(filename, configspec)