def get_or_create(cls, key, defaults={}):
        '''
        A port of functionality from the Django ORM. Defaults can be passed in
        if creating a new document is necessary. Keyword args are used to
        lookup the document. Returns a tuple of (object, created), where object
        is the retrieved or created object and created is a boolean specifying
        whether a new object was created.
        '''
        instance = cls.get(key)
        if not instance:
            created = True
            data = dict(key=key)
            data.update(defaults)
            # Do an upsert here instead of a straight create to avoid a race
            # condition with another instance creating the same record at
            # nearly the same time.
            instance = cls.update(data, data, upsert=True)
        else:
            created = False
        return instance, created