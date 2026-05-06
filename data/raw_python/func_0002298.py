def plugins(self):
        """
        Get the set of plugins that this field may display.
        """
        from fluent_contents import extensions
        if self._plugins is None:
            return extensions.plugin_pool.get_plugins()
        else:
            try:
                return extensions.plugin_pool.get_plugins_by_name(*self._plugins)
            except extensions.PluginNotFound as e:
                raise extensions.PluginNotFound(str(e) + " Update the plugin list of '{0}.{1}' field or FLUENT_CONTENTS_PLACEHOLDER_CONFIG['{2}'] setting.".format(self.model._meta.object_name, self.name, self.slot))