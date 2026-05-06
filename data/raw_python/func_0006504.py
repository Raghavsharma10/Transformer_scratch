def recherche(self, pattern, entete, in_all=False):
        """abstractSearch in fields of collection and reset rendering.
        Returns number of results.
        If in_all is True, call get_all before doing the search."""
        if in_all:
            self.collection = self.get_all()
        self.collection.recherche(pattern, entete)
        self._reset_render()
        return len(self.collection)