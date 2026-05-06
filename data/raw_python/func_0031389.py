def select(self, fields=['rowid', '*'], offset=None, limit=None):
        ''' return SELECT SQL
        '''
        # base SQL
        SQL = 'SELECT %s FROM %s' % (','.join(fields), self._table)
        
        # selectors
        if self._selectors:
            SQL = ' '.join([SQL, 'WHERE', self._selectors]).strip()
        
        # modifiers
        if self._modifiers:
            SQL = ' '.join([SQL, self._modifiers])

        # limit
        if limit is not None and isinstance(limit, int):
            SQL = ' '.join((SQL, 'LIMIT %s' % limit))

        # offset
        if (limit is not None) and (offset is not None) and isinstance(offset, int):
            SQL = ' '.join((SQL, 'OFFSET %s' % offset))

        return ''.join((SQL, ';'))