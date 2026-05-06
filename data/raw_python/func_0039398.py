def show_tables(self, format='json'):
        '''Return list of all available tables'''

        query = 'SHOW TABLES'
        payload = self._payload_builder(query, format) 	

        response = self.execute_query(payload) 

        return response