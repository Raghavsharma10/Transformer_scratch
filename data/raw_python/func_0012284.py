def remove_option(self, section, option):
        """Remove an option.

        Args:
            section (str): section name
            option (str): option name

        Returns:
            bool: whether the option was actually removed
        """
        try:
            section = self.__getitem__(section)
        except KeyError:
            raise NoSectionError(section) from None
        option = self.optionxform(option)
        existed = option in section.options()
        if existed:
            del section[option]
        return existed