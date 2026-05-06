def _execute_query(self):
        '''Execute the query without fetching data. Returns the number of
elements in the query.'''
        pipe = self.pipe
        if not self.card:
            if self.meta.ordering:
                self.ismember = getattr(self.backend.client, 'zrank')
                self.card = getattr(pipe, 'zcard')
                self._check_member = self.zism
            else:
                self.ismember = getattr(self.backend.client, 'sismember')
                self.card = getattr(pipe, 'scard')
                self._check_member = self.sism
        else:
            self.ismember = None
        self.card(self.query_key)
        result = yield pipe.execute()
        yield result[-1]