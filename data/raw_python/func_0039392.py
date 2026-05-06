def get(self, *args, **kwargs):
        '''Just a select which returns a response
        >>> yql.get("geo.countries', ['name', 'woeid'], 5")
        '''
        self = self.select(*args, **kwargs)

        payload = self._payload_builder(self._query)
        response = self.execute_query(payload)

        return response