def register(self, plugin):
        """
        Make a plugin known to the CMS.

        :param plugin: The plugin class, deriving from :class:`ContentPlugin`.
        :type plugin: :class:`ContentPlugin`

        The plugin will be instantiated once, just like Django does this with :class:`~django.contrib.admin.ModelAdmin` classes.
        If a plugin is already registered, this will raise a :class:`PluginAlreadyRegistered` exception.
        """
        # Duck-Typing does not suffice here, avoid hard to debug problems by upfront checks.
        assert issubclass(plugin, ContentPlugin), "The plugin must inherit from `ContentPlugin`"
        assert plugin.model, "The plugin has no model defined"
        assert issubclass(plugin.model, ContentItem), "The plugin model must inherit from `ContentItem`"
        assert issubclass(plugin.form, ContentItemForm), "The plugin form must inherit from `ContentItemForm`"

        name = plugin.__name__  # using class here, no instance created yet.
        name = name.lower()
        if name in self.plugins:
            raise PluginAlreadyRegistered("{0}: a plugin with this name is already registered".format(name))

        # Avoid registering 2 plugins to the exact same model. If you want to reuse code, use proxy models.
        if plugin.model in self._name_for_model:
            # Having 2 plugins for one model breaks ContentItem.plugin and the frontend code
            # that depends on using inline-model names instead of plugins. Good luck fixing that.
            # Better leave the model==plugin dependency for now.
            existing_plugin = self.plugins[self._name_for_model[plugin.model]]
            raise ModelAlreadyRegistered("Can't register the model {0} to {2}, it's already registered to {1}!".format(
                plugin.model.__name__,
                existing_plugin.name,
                plugin.__name__
            ))

        # Make a single static instance, similar to ModelAdmin.
        plugin_instance = plugin()
        self.plugins[name] = plugin_instance
        self._name_for_model[plugin.model] = name       # Track reverse for model.plugin link

        # Only update lazy indexes if already created
        if self._name_for_ctype_id is not None:
            self._name_for_ctype_id[plugin.type_id] = name

        return plugin