def dock_json_get_plugin_conf(self, plugin_type, plugin_name):
        """
        Return the configuration for a plugin.

        Raises KeyError if there are no plugins of that type.
        Raises IndexError if the named plugin is not listed.
        """
        match = [x for x in self.dock_json[plugin_type] if x.get('name') == plugin_name]
        return match[0]