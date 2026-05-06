def items(self, section=_UNSET):
        """Return a list of (name, value) tuples for options or sections.

        If section is given, return a list of tuples with (name, value) for
        each option in the section. Otherwise, return a list of tuples with
        (section_name, section_type) for each section.

        Args:
            section (str): optional section name, default UNSET

        Returns:
            list: list of :class:`Section` or :class:`Option` objects
        """
        if section is _UNSET:
            return [(sect.name, sect) for sect in self.sections_blocks()]

        section = self.__getitem__(section)
        return [(opt.key, opt) for opt in section.option_blocks()]