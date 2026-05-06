def delete(self, response, **kwargs):
        '''
        If a record matching the instance id exists in the database, delete it.
        '''
        response_cls = self._get_instance(**kwargs)
        if response_cls:
            _action_and_commit(response_cls, session.delete)