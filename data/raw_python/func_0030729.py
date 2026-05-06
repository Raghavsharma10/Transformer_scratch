def create(self):
        """Create the database from the base SQL."""

        if not self.exists():
            self._create_path()
            self.create_tables()
            return True

        return False