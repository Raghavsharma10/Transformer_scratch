def plugins(self):
        """
        Get the set of plugins that this widget should display.
        """
        from fluent_contents import extensions   # Avoid circular reference because __init__.py imports subfolders too
        if self._plugins is None:
            return extensions.plugin_pool.get_plugins()
        else:
            return extensions.plugin_pool.get_plugins_by_name(*self._plugins)