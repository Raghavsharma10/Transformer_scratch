def register_entity(self, entity_config):
        """
        Registers an entity config
        """
        if not issubclass(entity_config, EntityConfig):
            raise ValueError('Must register entity config class of subclass EntityConfig')

        if entity_config.queryset is None:
            raise ValueError('Entity config must define queryset')

        model = entity_config.queryset.model

        self._entity_registry[model] = entity_config()

        # Add watchers to the global look up table
        for watching_model, entity_model_getter in entity_config.watching:
            self._entity_watching[watching_model].append((model, entity_model_getter))