def drop(self):
        """Drop the table from the database
        """
        if self._is_dropped is False:
            self.table.drop(self.engine)
        self._is_dropped = True