def has_option(self, section, option):
        """Checks for the existence of a given option in a given section.

        Args:
            section (str): name of section
            option (str): name of option

        Returns:
            bool: whether the option exists in the given section
        """
        if section not in self.sections():
            return False
        else:
            option = self.optionxform(option)
            return option in self[section]