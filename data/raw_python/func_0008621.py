def has_conversion(self, plugin):
        """Return True if the plugin supports this block."""
        plugin = kurt.plugin.Kurt.get_plugin(plugin)
        return plugin.name in self._plugins