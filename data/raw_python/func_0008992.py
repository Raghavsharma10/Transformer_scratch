def select_config_schema_for(cls, section_name):
        """Select the config schema that matches the config section (by name).

        :param section_name:    Config section name (as key).
        :return: Config section schmema to use (subclass of: SectionSchema).
        """
        # pylint: disable=cell-var-from-loop, redefined-outer-name
        for section_schema in cls.config_section_schemas:
            schema_matches = getattr(section_schema, "matches_section", None)
            if schema_matches is None:
                # -- OTHER SCHEMA CLASS: Reuse SectionSchema functionality.
                schema_matches = lambda name: SectionSchema.matches_section(
                    name, section_schema.section_names)

            if schema_matches(section_name):
                return section_schema
        return None