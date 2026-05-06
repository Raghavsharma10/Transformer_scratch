def delete_key(self, key_id):
        """Delete the key with the specified id from your deploy keys list.

        :returns: bool -- True if successful, False otherwise
        """
        if int(key_id) <= 0:
            return False
        url = self._build_url('keys', str(key_id), base_url=self._api)
        return self._boolean(self._delete(url), 204, 404)