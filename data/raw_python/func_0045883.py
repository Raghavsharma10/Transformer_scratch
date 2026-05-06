def remove_option(self, section, name, value=None):
        """Remove an option from a unit

        Args:
            section (str): The section to remove from.
            name (str): The item to remove.
            value (str, optional): If specified, only the option matching this value will be removed
                                   If not specified, all options with ``name`` in ``section`` will be removed

        Returns:
            True: At least one item was removed
            False: The item requested to remove was not found

        """
        # Don't allow updating units we loaded from fleet, it's not supported
        if self._is_live():
            raise RuntimeError('Submitted units cannot update their options')

        removed = 0
        # iterate through a copy of the options
        for option in list(self._data['options']):
            # if it's in our section
            if option['section'] == section:
                # and it matches our name
                if option['name'] == name:
                    # and they didn't give us a value, or it macthes
                    if value is None or option['value'] == value:
                        # nuke it from the source
                        self._data['options'].remove(option)
                        removed += 1

        if removed > 0:
            return True

        return False