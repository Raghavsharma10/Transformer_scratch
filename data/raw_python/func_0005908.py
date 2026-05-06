def derive_from(cls, section, name=None):
        """Creates a new section based on the given.

        :param Section section: Section to derive from,

        :param str|unicode name: New section name.

        :rtype: Section
        """
        new_section = deepcopy(section)

        if name:
            new_section.name = name

        return new_section