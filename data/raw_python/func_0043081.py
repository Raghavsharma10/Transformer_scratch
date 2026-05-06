def _find_config(self):
        """Searches through the configured `config_paths` for the `config_name`
        file.

        If there are no `config_paths` defined, this will raise an error, so the
        caller should take care to check the value of `config_paths` first.

        Returns:
            str: The fully qualified path to the configuration that was found.

        Raises:
            Exception: No paths are defined in `config_paths` or no file with
                the `config_name` was found in any of the specified `config_paths`.
        """
        for search_path in self.config_paths:
            for ext in self._fmt_to_ext.get(self.config_format):
                path = os.path.abspath(os.path.join(search_path, self.config_name + ext))
                if os.path.isfile(path):
                    self.config_file = path
                    return
        raise BisonError('No file named {} found in search paths {}'.format(
            self.config_name, self.config_paths))