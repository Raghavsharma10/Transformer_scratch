def get_assistants_from_file_hierarchy(cls, file_hierarchy, superassistant,
                                           role=settings.DEFAULT_ASSISTANT_ROLE):
        """Accepts file_hierarch as returned by cls.get_assistant_file_hierarchy and returns
        instances of YamlAssistant for loaded files

        Args:
            file_hierarchy: structure as described in cls.get_assistants_file_hierarchy
            role: role of all assistants in this hierarchy (we could find
                  this out dynamically but it's not worth the pain)
        Returns:
            list of top level assistants from given hierarchy; these assistants contain
            references to instances of their subassistants (and their subassistants, ...)
        """
        result = []
        warn_msg = 'Failed to load assistant {source}, skipping subassistants.'

        for name, attrs in file_hierarchy.items():
            loaded_yaml = yaml_loader.YamlLoader.load_yaml_by_path(attrs['source'])
            if loaded_yaml is None:  # there was an error parsing yaml
                logger.warning(warn_msg.format(source=attrs['source']))
                continue
            try:
                ass = cls.assistant_from_yaml(attrs['source'],
                                              loaded_yaml,
                                              superassistant,
                                              role=role)
            except exceptions.YamlError as e:
                logger.warning(e)
                continue
            ass._subassistants = cls.get_assistants_from_file_hierarchy(attrs['subhierarchy'],
                                                                        ass,
                                                                        role=role)
            result.append(ass)

        return result