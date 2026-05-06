def get_all_allowed_plugins(self):
        """
        Return which plugins are allowed by the placeholder fields.
        """
        # Get all allowed plugins of the various placeholders together.
        if not hasattr(self.model, '_meta_placeholder_fields'):
            # No placeholder fields in the model, no need for inlines.
            return []

        plugins = []
        for name, field in self.model._meta_placeholder_fields.items():
            assert isinstance(field, PlaceholderField)
            if field.plugins is None:
                # no limitations, so all is allowed
                return extensions.plugin_pool.get_plugins()
            else:
                plugins += field.plugins

        return list(set(plugins))