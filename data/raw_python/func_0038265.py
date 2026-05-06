def insert_many(self, table, columns, values, limit=MAX_ROWS_PER_QUERY, execute=True):
        """
        Insert multiple rows into a table.

        If only one row is found, self.insert method will be used.
        """
        # Make values a list of lists if it is a flat list
        if not isinstance(values[0], (list, set, tuple)):
            values = []
            for v in values:
                if v is not None and len(v) > 0:
                    values.append([v])
                else:
                    values.append([None])

        # Concatenate statement
        cols, vals = get_col_val_str(columns)
        statement = 'INSERT INTO {0} ({1}) VALUES ({2})'.format(wrap(table), cols, vals)

        if execute and len(values) > limit:
            while len(values) > 0:
                vals = [values.pop(0) for i in range(0, min(limit, len(values)))]
                self._cursor.executemany(statement, vals)
                self._commit()

        elif execute:
            # Execute statement
            self._cursor.executemany(statement, values)
            self._commit()
            self._printer('\tMySQL rows (' + str(len(values)) + ') successfully INSERTED')

        # Only return statement
        else:
            return statement