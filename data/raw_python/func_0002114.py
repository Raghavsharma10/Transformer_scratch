def get_version(self, version_id, expand=[]):
        """
        Get a specific version of this layer
        """
        target_url = self._client.get_url('VERSION', 'GET', 'single', {'layer_id': self.id, 'version_id': version_id})
        return self._manager._get(target_url, expand=expand)