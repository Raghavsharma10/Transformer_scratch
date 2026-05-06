def child(self, path):
        """
        Override the base implementation to inject the share ID our
        constructor was passed.
        """
        if self._shareID is not None:
            self = url.URL.child(self, self._shareID)
            self._shareID = None
        return url.URL.child(self, path)