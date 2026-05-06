def process_config_section(cls, config_section, storage):
        """Process the config section and store the extracted data in
        the param:`storage` (as outgoing param).
        """
        # -- CONCEPT:
        # if not storage:
        #     # -- INIT DATA: With default parts.
        #     storage.update(dict(_PERSONS={}))

        schema = cls.select_config_schema_for(config_section.name)
        if not schema:
            message = "No schema found for: section=%s"
            raise LookupError(message % config_section.name)

        # -- PARSE AND STORE CONFIG SECTION:
        section_storage = cls.select_storage_for(config_section.name, storage)
        section_data = parse_config_section(config_section, schema)
        section_storage.update(section_data)