def canonical_uri_path(self):
        """
        The canonicalized URI path from the request.
        """
        result = getattr(self, "_canonical_uri_path", None)
        if result is None:
            result = self._canonical_uri_path = get_canonical_uri_path(
                self.uri_path)
        return result