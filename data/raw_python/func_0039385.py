def raw_query(self, query, format=None, pretty=False):
        '''Executes a YQL query and returns a response
        >>>...
        >>> resp = yql.raw_query('select * from weather.forecast where woeid=2502265')
        >>>
        '''
        if format:
            format = format
        else:
            format = self.format

        payload = self._payload_builder(query, format=format)
        response = self.execute_query(payload)
        if pretty:
            response = self.response_builder(response)

        return response