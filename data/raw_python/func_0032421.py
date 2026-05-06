def query(self, sql, **kwargs):
        """
        Shortcut for GET /v1/query, except with argument format='dataframe'
        (the default), in which case it will simply wrap the result of the GET
        query to /v1/query (with format='table') in a `pandas.DataFrame`.
        """
        if 'format' not in kwargs or kwargs['format'] == 'dataframe':
            resp = self.get('/v1/query', data={'q': sql, 'format': 'table'}).json()
            if len(resp) == 0:
                return pd.DataFrame()
            else:
                return pd.DataFrame.from_records(resp[1:], columns=resp[0],
                                                 index="_rowName")
        kwargs['q'] = sql
        return self.get('/v1/query', **kwargs).json()