def get(self, section, option):
        """Gets an option value for a given section.

        Args:
            section (str): section name
            option (str): option name

        Returns:
            :class:`Option`: Option object holding key/value pair
        """
        if not self.has_section(section):
            raise NoSectionError(section) from None

        section = self.__getitem__(section)
        option = self.optionxform(option)
        try:
            value = section[option]
        except KeyError:
            raise NoOptionError(option, section)

        return value