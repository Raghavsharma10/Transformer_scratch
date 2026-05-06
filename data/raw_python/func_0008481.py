def append(self, entity, name="pers"):
        """ Appends a named entity to the lexicon,
            e.g., Entities.append("Hooloovoo", "PERS")
        """
        e = map(lambda s: s.lower(), entity.split(" ") + [name])
        self.setdefault(e[0], []).append(e)