def start_update(self, layer_id):
        """
        A shortcut to create a new version and start importing it.
        Effectively the same as :py:meth:`koordinates.layers.LayerManager.create_draft` followed by :py:meth:`koordinates.layers.LayerManager.start_import`.
        """
        target_url = self.client.get_url('LAYER', 'POST', 'update', {'layer_id': layer_id})
        r = self.client.request('POST', target_url, json={})
        return self.parent.create_from_result(r.json())