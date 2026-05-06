def _get_all(self):
        ''' return all items
        '''
        rowid = 0
        while True:
            SQL_SELECT_MANY = 'SELECT rowid, * FROM %s WHERE rowid > ? LIMIT ?;' % self._table
            self._cursor.execute(SQL_SELECT_MANY, (rowid, ITEMS_PER_REQUEST))
            items = self._cursor.fetchall()
            if len(items) == 0:
                break
            for item in items:
                rowid = item['_id']
                yield self._make_item(item)