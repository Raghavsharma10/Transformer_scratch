def append(self, term, type=None, value=None):
        """ Appends the given term to the taxonomy and tags it as the given type.
            Optionally, a disambiguation value can be supplied.
            For example: taxonomy.append("many", "quantity", "50-200")
        """
        term = self._normalize(term)
        type = self._normalize(type)
        self.setdefault(term, (odict(), odict()))[0].push((type, True))
        self.setdefault(type, (odict(), odict()))[1].push((term, True))
        self._values[term] = value