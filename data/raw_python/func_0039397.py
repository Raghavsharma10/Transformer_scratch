def where(self, *args):
        ''' This method simulates a where condition. Use as follow:
        >>> yql.select('mytable').where(['name', '=', 'alain'], ['location', '!=', 'paris'])
        '''
        if not self._table:
            raise errors.NoTableSelectedError('No Table Selected')

        clause = []
        self._query += ' WHERE '

        clause = [ self._clause_formatter(x) for x in args if x ]

        self._query += ' AND '.join(clause)

        payload = self._payload_builder(self._query)
        response = self.execute_query(payload)

        return response