def set(self, section, option, value=None):
        """Set an option.

        Args:
            section (str): section name
            option (str): option name
            value (str): value, default None
        """
        try:
            section = self.__getitem__(section)
        except KeyError:
            raise NoSectionError(section) from None
        option = self.optionxform(option)
        if option in section:
            section[option].value = value
        else:
            section[option] = value
        return self