def modify(self, SQL, params='', verbose=True):
        """
        Wrapper for CRUD operations to make them distinct from queries and automatically pass commit() method to cursor.

        Parameters
        ----------
        SQL: str
            The SQL query to execute
        params: sequence
            Mimics the native parameter substitution of sqlite3
        verbose: bool
                Prints the number of modified records
        """
        # Make sure the database isn't locked
        self.conn.commit()

        if SQL.lower().startswith('select'):
            print('Use self.query method for queries.')
        else:
            self.list(SQL, params)
            self.conn.commit()
            if verbose:
                print('Number of records modified: {}'.format(self.list("SELECT changes()").fetchone()[0] or '0'))