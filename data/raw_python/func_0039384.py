def _payload_builder(self, query, format=None):
        '''Build the payload'''
        if self.community :
            query = self.COMMUNITY_DATA + query # access to community data tables

        if vars(self).get('yql_table_url') : # Attribute only defined when MYQL.use has been called before
            query = "use '{0}' as {1}; ".format(self.yql_table_url, self.yql_table_name) + query

        if vars(self).get('_func'): # if post query function filters
            query = '| '.join((query, self._func))

        self._query = query        
        
        self._query = self._add_limit()
        self._query = self._add_offset()

        logger.info("QUERY = %s" %(self._query,))

        payload = {
            'q': self._query,
            'callback': '',#This is not javascript
            'diagnostics': self.diagnostics,
            'format': format if format else self.format,
            'debug': self.debug,
            'jsonCompact': 'new' if self.jsonCompact else ''
        }

        if vars(self).get('_vars'):
            payload.update(self._vars)

        if self.crossProduct:
            payload['crossProduct'] = 'optimized'

        self._payload = payload
        logger.info("PAYLOAD = %s " %(payload, ))

        return payload