def delete(self, criteria=None, _all=False):
        ''' delete dictionary(ies) in sqlite database

        _all = True - delete all items
        '''
        if isinstance(criteria, self._item_class):
            criteria = {'_id': criteria['_id']}

        if criteria is None and not _all:
            raise RuntimeError('Criteria is not defined')

        SQL = SQLBuilder(self._table, criteria).delete()
        self._cursor.execute(SQL)