def get_published(self, layer_id, expand=[]):
        """
        Get the latest published version of this layer.
        :raises NotFound: if there is no published version.
        """
        target_url = self.client.get_url('VERSION', 'GET', 'published', {'layer_id': layer_id})
        return self._get(target_url, expand=expand)