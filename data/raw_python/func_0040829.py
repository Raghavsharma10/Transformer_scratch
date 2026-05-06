def update(self, response, **kwargs):
        '''
        If a record matching the instance already exists in the database, update
        both the column and venue column attributes, else create a new record.
        '''
        response_cls = super(
            LocationResponseClassLegacyAccessor, self)._get_instance(**kwargs)
        if response_cls:
            setattr(response_cls, self.column, self.accessor(response))
            setattr(
                response_cls, self.venue_column, self.venue_accessor(response))
            _action_and_commit(response_cls, session.add)