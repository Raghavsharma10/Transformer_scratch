def _validate_sections(cls, sections):
        """Validates sections types and uniqueness."""
        names = []
        for section in sections:

            if not hasattr(section, 'name'):
                raise ConfigurationError('`sections` attribute requires a list of Section')

            name = section.name
            if name in names:
                raise ConfigurationError('`%s` section name must be unique' % name)

            names.append(name)