def select(self, table, items=None, limit=None, offset=None, remote_filter=None, func_filters=None):
        '''This method simulate a select on a table
        >>> yql.select('geo.countries', limit=5) 
        >>> yql.select('social.profile', ['guid', 'givenName', 'gender'])
        '''
        self._table = table

        if remote_filter:
            if not isinstance(remote_filter, tuple):
                raise TypeError("{0} must be of type <type tuple>".format(remote_filter))

            table = "%s(%s)" %(table, ','.join(map(str, remote_filter)))

        if not items:
            items = ['*']
        self._query = "SELECT {1} FROM {0} ".format(table, ','.join(items))

        if func_filters:
            self._func = self._func_filters(func_filters) 

        self._limit = limit
        self._offset = offset
            
        return self