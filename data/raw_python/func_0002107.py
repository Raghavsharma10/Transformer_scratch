def create_draft(self, layer_id):
        """
        Creates a new draft version.

        If anything in the data object has changed then an import will begin immediately.
        Otherwise to force a re-import from the previous sources call :py:meth:`koordinates.layers.LayerManager.start_import`.

        :rtype: Layer
        :return: the new version
        :raises Conflict: if there is already a draft version for this layer.
        """
        target_url = self.client.get_url('VERSION', 'POST', 'create', {'layer_id': layer_id})
        r = self.client.request('POST', target_url, json={})
        return self.create_from_result(r.json())