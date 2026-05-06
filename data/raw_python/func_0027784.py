def executeSQL(self, sql, args=()):
        """
        For use with UPDATE or INSERT statements.
        """
        sql = self._execSQL(sql, args)
        result = self.cursor.lastRowID()
        if self.executedThisTransaction is not None:
            self.executedThisTransaction.append((result, sql, args))
        return result