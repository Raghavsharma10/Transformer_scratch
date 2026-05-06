def _drop(self, table, existing_tables=None):
        """Private method for executing table drop commands."""
        # Retrieve list of existing tables for comparison
        existing_tables = existing_tables if existing_tables else self.tables

        # Only drop table if it exists
        if table in existing_tables:
            # Set to avoid foreign key errorrs
            self.execute('SET FOREIGN_KEY_CHECKS = 0')

            query = 'DROP TABLE {0}'.format(wrap(table))
            self.execute(query)

            # Set again
            self.execute('SET FOREIGN_KEY_CHECKS = 1')
            self._printer('\tDropped table {0}'.format(table))