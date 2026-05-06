def _validate_required(self, settings, name, value):
        """
        Validate a required setting (value can not be empty)

        Args:
            settings (dict): Current settings.
            name (str): Setting name.
            value (str): Required value to validate.

        Raises:
            boussole.exceptions.SettingsInvalidError: If value is empty.

        Returns:
            str: Validated value.

        """
        if not value:
            raise SettingsInvalidError(("Required value from setting '{name}' "
                                        "must not be "
                                        "empty.").format(name=name))

        return value