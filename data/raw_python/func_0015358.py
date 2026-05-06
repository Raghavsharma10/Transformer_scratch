def get_assistants_from_cache_hierarchy(cls, cache_hierarchy, superassistant,
                                            role=settings.DEFAULT_ASSISTANT_ROLE):
        """Accepts cache_hierarch as described in devassistant.cache and returns
        instances of YamlAssistant (only with cached attributes) for loaded files

        Args:
            cache_hierarchy: structure as described in devassistant.cache
            role: role of all assistants in this hierarchy (we could find
                  this out dynamically but it's not worth the pain)
        Returns:
            list of top level assistants from given hierarchy; these assistants contain
            references to instances of their subassistants (and their subassistants, ...)
            Note, that the assistants are not fully loaded, but contain just cached attrs.
        """
        result = []

        for name, attrs in cache_hierarchy.items():
            ass = cls.assistant_from_yaml(attrs['source'],
                                          {name: attrs['attrs']},
                                          superassistant,
                                          fully_loaded=False,
                                          role=role)
            ass._subassistants = cls.get_assistants_from_cache_hierarchy(attrs['subhierarchy'],
                                                                         ass,
                                                                         role=role)
            result.append(ass)

        return result