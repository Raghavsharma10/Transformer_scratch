def update(self, response, **kwargs):
        '''
        If a record matching the instance already exists in the database, update
        it, else create a new record.
        '''
        response_cls = self._get_instance(**kwargs)
        if response_cls:
            setattr(response_cls, self.column, self.accessor(response))
            _action_and_commit(response_cls, session.add)
        else:
            self.get_or_create_from_legacy_response(response, **kwargs)