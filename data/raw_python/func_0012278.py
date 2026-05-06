def add_section(self, section):
        """Create a new section in the configuration.

        Raise DuplicateSectionError if a section by the specified name
        already exists. Raise ValueError if name is DEFAULT.

        Args:
            section (str or :class:`Section`): name or Section type
        """
        if section in self.sections():
            raise DuplicateSectionError(section)
        if isinstance(section, str):
            # create a new section
            section = Section(section, container=self)
        elif not isinstance(section, Section):
            raise ValueError("Parameter must be a string or Section type!")
        self._structure.append(section)