def get_plugin(self, name):
        """
        Returns a plugin by name and raises ``gemstone.errors.PluginDoesNotExistError`` error if
        no plugin with such name exists.

        :param name: a string specifying a plugin name.
        :return: the corresponding plugin instance.
        """
        for plugin in self.plugins:
            if plugin.name == name:
                return plugin

        raise PluginDoesNotExistError("Plugin '{}' not found".format(name))