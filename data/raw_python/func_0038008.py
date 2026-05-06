def truncate_database(self, database=None):
        """Drop all tables in a database."""
        # Change database if needed
        if database in self.databases and database is not self.database:
            self.change_db(database)

        # Get list of tables
        tables = self.tables if isinstance(self.tables, list) else [self.tables]
        if len(tables) > 0:
            self.drop(tables)
            self._printer('\t' + str(len(tables)), 'tables truncated from', database)
        return tables