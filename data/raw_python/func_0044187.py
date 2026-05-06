def valid_paths(self, *args):
        """
        Validate that given paths are not the same.

        Args:
            (string): Path to validate.

        Raises:
            boussole.exceptions.SettingsInvalidError: If there is more than one
                occurence of the same path.

        Returns:
            bool: ``True`` if paths are validated.
        """
        for i, path in enumerate(args, start=0):
            cp = list(args)
            current = cp.pop(i)
            if current in cp:
                raise SettingsInvalidError("Multiple occurences finded for "
                                           "path: {}".format(current))

        return True