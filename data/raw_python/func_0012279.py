def options(self, section):
        """Returns list of configuration options for the named section.

        Args:
            section (str): name of section

        Returns:
            list: list of option names
        """
        if not self.has_section(section):
            raise NoSectionError(section) from None
        return self.__getitem__(section).options()