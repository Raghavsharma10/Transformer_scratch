def _apply_scope(self, config, tags):
        """Add locally scoped tags to config"""
        if isinstance(config, dict):
            # Recursively _apply_scope for each item in the config
            for val in config.values():
                self._apply_scope(val, tags)

        elif isinstance(config, list):
            # Recursively _apply_scope for each item in the config
            for item in config:
                self._apply_scope(item, tags)

        elif isinstance(config, TaggedValue):
            tagged_value = config

            # add all local tags
            tagged_value.tags |= set(tags)
            for tag in tags:
                if not hasattr(tagged_value, tag):
                    setattr(tagged_value, tag, False)