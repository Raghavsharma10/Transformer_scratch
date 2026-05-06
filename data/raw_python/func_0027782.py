def querySQL(self, sql, args=()):
        """For use with SELECT (or SELECT-like PRAGMA) statements.
        """
        if self.debug:
            result = timeinto(self.queryTimes, self._queryandfetch, sql, args)
        else:
            result = self._queryandfetch(sql, args)
        return result