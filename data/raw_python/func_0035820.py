def _insert(self, trigram):
        """
        Insert a trigram in the DB
        """
        words = list(map(self._sanitize, trigram))

        key = self._WSEP.join(words[:2]).lower()
        next_word = words[2]

        self._db.setdefault(key, [])
        # we could use a set here, but sets are not serializables in JSON. This
        # is the same reason we use dicts instead of defaultdicts.
        if next_word not in self._db[key]:
            self._db[key].append(next_word)