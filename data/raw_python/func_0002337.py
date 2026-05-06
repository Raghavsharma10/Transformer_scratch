def get_plugins_by_name(self, *names):
        """
        Return a list of plugins by plugin class, or name.
        """
        self._import_plugins()
        plugin_instances = []
        for name in names:
            if isinstance(name, six.string_types):
                try:
                    plugin_instances.append(self.plugins[name.lower()])
                except KeyError:
                    raise PluginNotFound("No plugin named '{0}'.".format(name))
            elif isinstance(name, type) and issubclass(name, ContentPlugin):
                # Will also allow classes instead of strings.
                plugin_instances.append(self.plugins[self._name_for_model[name.model]])
            else:
                raise TypeError("get_plugins_by_name() expects a plugin name or class, not: {0}".format(name))
        return plugin_instances