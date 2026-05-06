def _load(self):
        """
        Load the database from its ``dbfile`` if it has one
        """
        if self.dbfile is not None:
            with open(self.dbfile, 'r') as f:
                self._db = json.loads(f.read())
        else:
            self._db = {}