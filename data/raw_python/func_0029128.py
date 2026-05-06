def link(self, key1, key2):
        """
        Make these two keys have the same value
        :param key1:
        :param key2:
        :return:
        """
        # TODO make this have more than one key linked
        # TODO Maybe make the value a set?
        self._linked_keys[key1] = key2
        self._linked_keys[key2] = key1