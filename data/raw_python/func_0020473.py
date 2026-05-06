def get_plugin_conf(self, phase, name):
        """
        Return the configuration for a plugin.

        Raises KeyError if there are no plugins of that type.
        Raises IndexError if the named plugin is not listed.
        """
        match = [x for x in self.template[phase] if x.get('name') == name]
        return match[0]