def get_plugin_by_model(self, model_class):
        """
        Return the corresponding plugin for a given model.

        You can also use the :attr:`ContentItem.plugin <fluent_contents.models.ContentItem.plugin>` property directly.
        This is the low-level function that supports that feature.
        """
        self._import_plugins()                       # could happen during rendering that no plugin scan happened yet.
        assert issubclass(model_class, ContentItem)  # avoid confusion between model instance and class here!

        try:
            name = self._name_for_model[model_class]
        except KeyError:
            raise PluginNotFound("No plugin found for model '{0}'.".format(model_class.__name__))
        return self.plugins[name]