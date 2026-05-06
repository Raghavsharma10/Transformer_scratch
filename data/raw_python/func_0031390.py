def delete(self):
        ''' return DELETE SQL
        '''
        SQL = 'DELETE FROM %s' % self._table
        if self._selectors:
            SQL = ' '.join([SQL, 'WHERE', self._selectors]).strip()
        
        return SQL