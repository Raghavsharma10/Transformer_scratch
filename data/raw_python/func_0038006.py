def truncate(self, table):
        """Empty a table by deleting all of its rows."""
        if isinstance(table, (list, set, tuple)):
            for t in table:
                self._truncate(t)
        else:
            self._truncate(table)