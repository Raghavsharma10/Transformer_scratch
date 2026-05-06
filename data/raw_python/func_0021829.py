def get_plugin(self, method):
        """
        Return plugin object if CLI option is activated and method exists

        @param method: name of plugin's method we're calling
        @type method: string

        @returns: list of plugins with `method`

        """
        all_plugins = []
        for entry_point in pkg_resources.iter_entry_points('yolk.plugins'):
            plugin_obj = entry_point.load()
            plugin = plugin_obj()
            plugin.configure(self.options, None)
            if plugin.enabled:
                if not hasattr(plugin, method):
                    self.logger.warn("Error: plugin has no method: %s" % method)
                    plugin = None
                else:
                    all_plugins.append(plugin)
        return all_plugins