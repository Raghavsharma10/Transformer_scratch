def delete_version(self, version_id=None):
        """
        Deletes this draft version (revert to published)

        :raises NotAllowed: if this version is already published.
        :raises Conflict: if this version is already deleted.
        """
        if not version_id:
            version_id = self.version.id

        target_url = self._client.get_url('VERSION', 'DELETE', 'single', {'layer_id': self.id, 'version_id': version_id})
        r = self._client.request('DELETE', target_url)
        logger.info("delete_version(): %s", r.status_code)