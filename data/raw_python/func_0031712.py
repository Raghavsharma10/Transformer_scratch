def get_config(self):
        """
        Load user configuration or return default when not found.

        :rtype: :class:`Configuration`

        """
        if not self._config:
            namespace = {}
            if os.path.exists(self.config_path):
                execfile(self.config_path, namespace)
            self._config = namespace.get('config') or Configuration()
        return self._config