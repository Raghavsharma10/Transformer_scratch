def use(self, plugin, arguments={}):
    """Add plugin to use during compilation.

    plugin: Plugin to include.
    arguments: Dictionary of arguments to pass to the import.
    """
    self.plugins[plugin] = dict(arguments)
    return self.plugins