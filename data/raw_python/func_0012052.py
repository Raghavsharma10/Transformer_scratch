def create(self):
        """Create the corresponding index. Will overwrite existing indexes of the same name."""
        body = dict()
        if self.mapping is not None:
            body['mappings'] = self.mapping
        if self.settings is not None:
            body['settings'] = self.settings
        else:
            body['settings'] = self._default_settings()
        self.instance.indices.create(self.index, body)