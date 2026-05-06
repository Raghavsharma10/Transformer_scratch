def commit(self, callback=None):
        '''Close the transaction and commit session to the backend.'''
        if self.executed:
            raise InvalidTransaction('Invalid operation. '
                                     'Transaction already executed.')
        session = self.session
        self.session = None
        self.on_result = self._commit(session, callback)
        return self.on_result