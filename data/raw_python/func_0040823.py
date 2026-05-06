def delete(cls, **kwargs):
        '''
        If a record matching the instance id exists in the database, delete it.
        '''
        q = cls._get_instance(**kwargs)
        if q:
            _action_and_commit(q, session.delete)