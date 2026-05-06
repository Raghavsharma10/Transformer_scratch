def _get_with_criteria(self, criteria, offset=None, limit=None):
        ''' returns items selected by criteria
        '''
        SQL = SQLBuilder(self._table, criteria).select(offset=offset, limit=limit)
        self._cursor.execute(SQL)
        for item in self._cursor.fetchall():
            yield self._make_item(item)