def get_draft(self, layer_id, expand=[]):
        """
        Get the current draft version of a layer.
        :raises NotFound: if there is no draft version.
        """
        target_url = self.client.get_url('VERSION', 'GET', 'draft', {'layer_id': layer_id})
        return self._get(target_url, expand=expand)