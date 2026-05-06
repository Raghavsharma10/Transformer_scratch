def _validate_path(self, settings, name, value):
        """
        Validate path exists

        Args:
            settings (dict): Current settings.
            name (str): Setting name.
            value (str): Path to validate.

        Raises:
            boussole.exceptions.SettingsInvalidError: If path does not exists.

        Returns:
            str: Validated path.

        """
        if not os.path.exists(value):
            raise SettingsInvalidError("Path from setting '{name}' does not "
                                       "exists: {value}".format(
                                           name=name,
                                           value=value
                                       ))

        return value