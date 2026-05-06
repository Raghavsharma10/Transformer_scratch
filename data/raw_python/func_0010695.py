def load_from_dict(self, conf_dict=None):
        """ Load the configuration from a dictionary.

        Args:
            conf_dict (dict): Dictionary with the configuration.
        """
        self.set_to_default()
        self._update_dict(self._config, conf_dict)
        self._update_python_paths()