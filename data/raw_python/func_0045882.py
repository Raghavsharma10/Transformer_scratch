def add_option(self, section, name, value):
        """Add an option to a section of the unit file

        Args:
            section (str): The name of the section, If it doesn't exist it will be created
            name (str): The name of the option to add
            value (str): The value of the option

        Returns:
            True: The item was added

        """

        # Don't allow updating units we loaded from fleet, it's not supported
        if self._is_live():
            raise RuntimeError('Submitted units cannot update their options')

        option = {
            'section': section,
            'name': name,
            'value': value
        }

        self._data['options'].append(option)

        return True