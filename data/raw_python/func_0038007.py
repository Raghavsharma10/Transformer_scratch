def _truncate(self, table):
        """
        Remove all records from a table in MySQL

        It performs the same function as a DELETE statement without a WHERE clause.
        """
        statement = "TRUNCATE TABLE {0}".format(wrap(table))
        self.execute(statement)
        self._printer('\tTruncated table {0}'.format(table))