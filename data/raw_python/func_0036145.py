def get_config(self, hostname):
        """
        Returns a configuration for hostname.

        """
        version, config = self._get(
            self.associations.get(hostname)
        )
        return config