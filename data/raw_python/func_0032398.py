def search(self, term, keywords=None, sortAscending=True):
        """
        Search the database.
        """
        if sortAscending:
            direction = 'ASC'
        else:
            direction = 'DESC'

        return [_SQLiteResultWrapper(r[0]) for r in
                self.store.querySQL(self.searchSQL % (direction,), (term,))]