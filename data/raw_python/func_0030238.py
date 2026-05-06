def index(self, prefix):
        """
        Return the model index for a prefix.
        """
        # Any web domain will be handled by the standard URLField.
        if self.is_external_url_prefix(prefix):
            prefix = 'http'

        for i, urltype in enumerate(self._url_types):
            if urltype.prefix == prefix:
                return i
        return None