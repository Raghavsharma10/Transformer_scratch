def get_draft_version(self, expand=[]):
        """
        Get the current draft version of this layer.
        :raises NotFound: if there is no draft version.
        """
        target_url = self._client.get_url('VERSION', 'GET', 'draft', {'layer_id': self.id})
        return self._manager._get(target_url, expand=expand)