def set(self, uri, content):
        """
        Cache node content for uri.
        No return.
        """
        key, value = self._prepare_node(uri, content)
        self._set(key, value)