def count(self):
        """
        Count the number of distinct results of the wrapped query.

        @return: an L{int} representing the number of distinct results.
        """
        if not self.query.store.autocommit:
            self.query.store.checkpoint()
        target = ', '.join([
            tableClass.storeID.getColumnName(self.query.store)
            for tableClass in self.query.tableClass ])
        sql, args = self.query._sqlAndArgs(
            'SELECT DISTINCT',
            target)
        sql = 'SELECT COUNT(*) FROM (' + sql + ')'
        result = self.query.store.querySQL(sql, args)
        assert len(result) == 1, 'more than one result: %r' % (result,)
        return result[0][0] or 0