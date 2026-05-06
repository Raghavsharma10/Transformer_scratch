def has_plugin_conf(self, phase, name):
        """
        Check whether a plugin is configured.
        """
        try:
            self.get_plugin_conf(phase, name)
            return True
        except (KeyError, IndexError):
            return False