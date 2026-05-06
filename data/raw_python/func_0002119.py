def save(self, with_data=False):
        """
        Edits this draft layerversion.
        # If anything in the data object has changed, cancel any existing import and start a new one.

        :param bool with_data: if ``True``, send the data object, which will start a new import and cancel
            any existing one. If ``False``, the data object will *not* be sent, and no import will start.
        :raises NotAllowed: if the version is already published.
        """
        target_url = self._client.get_url('VERSION', 'PUT', 'edit', {'layer_id': self.id, 'version_id': self.version.id})
        r = self._client.request('PUT', target_url, json=self._serialize(with_data=with_data))
        return self._deserialize(r.json(), self._manager)