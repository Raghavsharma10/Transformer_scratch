def section(self, section):
        """Creates a section block

        Args:
            section (str or :class:`Section`): name of section or object

        Returns:
            self for chaining
        """
        if not isinstance(self._container, ConfigUpdater):
            raise ValueError("Sections can only be added at section level!")
        if isinstance(section, str):
            # create a new section
            section = Section(section, container=self._container)
        elif not isinstance(section, Section):
            raise ValueError("Parameter must be a string or Section type!")
        if section.name in [block.name for block in self._container
                            if isinstance(block, Section)]:
            raise DuplicateSectionError(section.name)
        self._container.structure.insert(self._idx, section)
        self._idx += 1
        return self