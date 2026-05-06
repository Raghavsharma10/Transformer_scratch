def get_hash(self, keys=[]):
        """Return a unique hash associated with the listed keys."""
        if not len(keys):
            keys = list(self.keys())

        string_rep = ''
        oself = self._ordered(deepcopy(self))
        for key in keys:
            string_rep += json.dumps(oself.get(key, ''), sort_keys=True)

        return hashlib.sha512(string_rep.encode()).hexdigest()[:16]