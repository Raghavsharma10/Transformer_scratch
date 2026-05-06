def set(self, uri, content, **meta):
        """
        Dispatches private update/create handlers
        """
        try:
            node = self._update(uri, content, **meta)
            created = False
        except NodeDoesNotExist:
            node = self._create(uri, content, **meta)
            created = True
        return self._serialize(uri, node), created