def update(cls, **kwargs):
        '''
        If a record matching the instance id already exists in the database, 
        update it. If a record matching the instance id does not already exist,
        create a new record.
        '''
        q = cls._get_instance(**{'id': kwargs['id']})
        if q:
            for k, v in kwargs.items():
                setattr(q, k, v)
            _action_and_commit(q, session.add)
        else:
            cls.get_or_create(**kwargs)