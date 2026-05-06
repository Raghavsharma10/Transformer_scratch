def TENSES(self):
        """ Yields a list of tenses for this language, excluding negations.
            Each tense is a (tense, person, number, mood, aspect)-tuple.
        """
        a = set(TENSES[id] for id in self._format)
        a = a.union(set(TENSES[id] for id in self._default.keys()))
        a = a.union(set(TENSES[id] for id in self._default.values()))
        a = sorted(x[:-2] for x in a if x[-2] is False) # Exclude negation.
        return a