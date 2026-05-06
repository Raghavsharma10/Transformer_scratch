def head(self, n=10, by=None, **kwargs):
        """
        Get the first n entries for a given Table/Column. Additional keywords
        passed to QueryDb.query().

        Requires that the given table has a primary key specified.
        """
        col, id_col = self._query_helper(by=by)

        select = ("SELECT %s FROM %s ORDER BY %s ASC LIMIT %d" %
                  (col, self.table.name, id_col, n))

        return self._db.query(select, **kwargs)