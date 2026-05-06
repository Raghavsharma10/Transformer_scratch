def _list_keys(self):
        """
        Retrieves a list of all added Keys and populates the
        self._keys dict with Key instances

        :returns: A list of Keys instances
        """
        req = self.request(self.uri + '/keys')
        keys = req.get().json()
        if keys:
            self._keys = {}
            for key in keys:
                self._keys[key['id']] = Key(key, self)
        else:
            self._keys = {}