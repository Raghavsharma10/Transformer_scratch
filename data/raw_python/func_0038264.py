def insert(self, table, columns, values, execute=True):
        """Insert a single row into a table."""
        # TODO: Cant accept lists?
        # Concatenate statement
        cols, vals = get_col_val_str(columns)
        statement = "INSERT INTO {0} ({1}) VALUES ({2})".format(wrap(table), cols, vals)

        # Execute statement
        if execute:
            self._cursor.execute(statement, values)
            self._commit()
            self._printer('\tMySQL row successfully inserted into {0}'.format(table))

        # Only return statement
        else:
            return statement