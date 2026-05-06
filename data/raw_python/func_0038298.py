def _fetch(self, statement, commit, max_attempts=5):
        """
        Execute a SQL query and return a result.

        Recursively disconnect and reconnect to the database
        if an error occurs.
        """
        if self._auto_reconnect:
            attempts = 0
            while attempts < max_attempts:
                try:
                    # Execute statement
                    self._cursor.execute(statement)
                    fetch = self._cursor.fetchall()
                    rows = self._fetch_rows(fetch)
                    if commit:
                        self._commit()

                    # Return a single item if the list only has one item
                    return rows[0] if len(rows) == 1 else rows
                except Exception as e:
                    if attempts >= max_attempts:
                        raise e
                    else:
                        attempts += 1
                        self.reconnect()
                        continue
        else:
            # Execute statement
            self._cursor.execute(statement)
            fetch = self._cursor.fetchall()
            rows = self._fetch_rows(fetch)
            if commit:
                self._commit()

            # Return a single item if the list only has one item
            return rows[0] if len(rows) == 1 else rows