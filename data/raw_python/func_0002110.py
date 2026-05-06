def set_metadata(self, layer_id, version_id, fp):
        """
        Set the XML metadata on a layer draft version.

        :param file fp: file-like object to read the XML metadata from.
        :raises NotAllowed: if the version is already published.
        """
        base_url = self.client.get_url('VERSION', 'GET', 'single', {'layer_id': layer_id, 'version_id': version_id})
        self._metadata.set(base_url, fp)