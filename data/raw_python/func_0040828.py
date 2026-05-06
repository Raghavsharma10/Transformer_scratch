def get_or_create_from_legacy_response(self, response, **kwargs):
        '''
        If a record matching the instance already does not already exist in the
        database, then create a new record.
        '''
        response_cls = self.response_class(**kwargs).get_or_create(**kwargs)
        if not getattr(response_cls, self.column):
            setattr(response_cls, self.column, self.accessor(response))
            _action_and_commit(response_cls, session.add)
        if not getattr(response_cls, self.venue_column):
            setattr(
                response_cls, self.venue_column, self.venue_accessor(response))
            _action_and_commit(response_cls, session.add)