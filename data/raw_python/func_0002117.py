def start_update(self):
        """
        A shortcut to create a new version and start importing it.
        Effectively the same as :py:meth:`.create_draft_version` followed by :py:meth:`koordinates.layers.Layer.start_import`.

        :rtype: Layer
        :return: the new version
        :raises Conflict: if there is already a draft version for this layer.
        """
        target_url = self._client.get_url('LAYER', 'POST', 'update', {'layer_id': self.id})
        r = self._client.request('POST', target_url, json={})
        return self._manager.create_from_result(r.json())