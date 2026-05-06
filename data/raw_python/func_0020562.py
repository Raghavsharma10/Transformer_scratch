def dock_json_has_plugin_conf(self, plugin_type, plugin_name):
        """
        Check whether a plugin is configured.
        """

        try:
            self.dock_json_get_plugin_conf(plugin_type, plugin_name)
            return True
        except (KeyError, IndexError):
            return False