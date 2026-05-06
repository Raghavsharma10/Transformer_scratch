def _validate_paths(self, settings, name, value):
        """
        Apply ``SettingsPostProcessor._validate_path`` to each element in
        list.

        Args:
            settings (dict): Current settings.
            name (str): Setting name.
            value (list): List of paths to patch.

        Raises:
            boussole.exceptions.SettingsInvalidError: Once a path does not
                exists.

        Returns:
            list: Validated paths.

        """
        return [self._validate_path(settings, name, item)
                for item in value]