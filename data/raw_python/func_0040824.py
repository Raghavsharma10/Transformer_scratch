def _get_instance(self, **kwargs):
        '''Return the first existing instance of the response record.
        '''
        return session.query(self.response_class).filter_by(**kwargs).first()