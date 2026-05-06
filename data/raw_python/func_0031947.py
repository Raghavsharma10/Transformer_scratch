def _init_db(self):
        """Creates the database tables."""
        with self._get_db() as db:
            with open(self.schemapath) as f:
                db.cursor().executescript(f.read())
            db.commit()