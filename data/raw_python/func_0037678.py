def contents(self):
        """performs .get_content() and caches it in a ._content property
        """
        if not hasattr(self, "_content"):
            self._content = self.get_content()
        return self._content