def convert(self, plugin=None):
        """Return a :class:`PluginBlockType` for the given plugin name.

        If plugin is ``None``, return the first registered plugin.

        """
        if plugin:
            plugin = kurt.plugin.Kurt.get_plugin(plugin)
            if plugin.name in self._plugins:
                return self._plugins[plugin.name]
            else:
                err = BlockNotSupported("%s doesn't have %r" %
                        (plugin.display_name, self))
                err.block_type = self
                raise err
        else:
            return self.conversions[0]