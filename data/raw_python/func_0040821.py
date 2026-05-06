def get_or_create(cls, **kwargs):
        '''
        If a record matching the instance already exists in the database, then
        return it, otherwise create a new record.
        '''
        q = cls._get_instance(**kwargs)
        if q:
            return q
        q = cls(**kwargs)
        _action_and_commit(q, session.add)
        return q