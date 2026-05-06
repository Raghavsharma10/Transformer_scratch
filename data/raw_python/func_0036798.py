def keys(self, id=None, search=None):
        """
        Property-like function to call the _list_keys function in
        order to populate self._keys dict

        :returns: A list of Key instances
        """
        if self._keys is None:
            self._keys = {}
            self._list_keys()

        if id:
            return [self._keys[key_id] for key_id in self._keys.keys()
                    if id == self._keys[key_id].id]
        elif search:
            return [self._keys[key_id] for key_id in self._keys.keys()
                    if (search in self._keys[key_id].id) or (search in self._keys[key_id].name)]
        else:
            return [self._keys[key_id] for key_id in self._keys.keys()]