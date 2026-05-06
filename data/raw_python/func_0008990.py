def collect_config_sections_from_schemas(cls, config_section_schemas=None):
        # pylint: disable=invalid-name
        """Derive support config section names from config section schemas.
        If no :param:`config_section_schemas` are provided, the schemas from
        this class are used (normally defined in the DerivedClass).

        :param config_section_schemas:  List of config section schema classes.
        :return: List of config section names or name patterns (as string).
        """
        if config_section_schemas is None:
            config_section_schemas = cls.config_section_schemas

        collected = []
        for schema in config_section_schemas:
            collected.extend(schema.section_names)
            # -- MAYBE BETTER:
            # for name in schema.section_names:
            #    if name not in collected:
            #        collected.append(name)
        return collected