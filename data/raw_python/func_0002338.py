def get_model_classes(self):
        """
        Return all :class:`~fluent_contents.models.ContentItem` model classes which are exposed by plugins.
        """
        self._import_plugins()
        return [plugin.model for plugin in self.plugins.values()]