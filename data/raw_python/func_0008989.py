def matches_section(cls, section_name, supported_section_names=None):
        """Indicates if this schema can be used for a config section
        by using the section name.

        :param section_name:    Config section name to check.
        :return: True, if this schema can be applied to the config section.
        :return: Fals, if this schema does not match the config section.
        """
        if supported_section_names is None:
            supported_section_names = getattr(cls, "section_names", None)

        # pylint: disable=invalid-name
        for supported_section_name_or_pattern in supported_section_names:
            if fnmatch(section_name, supported_section_name_or_pattern):
                return True
        # -- OTHERWISE:
        return False