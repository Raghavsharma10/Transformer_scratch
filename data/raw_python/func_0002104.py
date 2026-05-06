def get_version(self, layer_id, version_id, expand=[]):
        """
        Get a specific version of a layer.
        """
        target_url = self.client.get_url('VERSION', 'GET', 'single', {'layer_id': layer_id, 'version_id': version_id})
        return self._get(target_url, expand=expand)