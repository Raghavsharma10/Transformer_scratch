def create_from_key(cls, oldvalue, key):
    "Raise  if the old value already exists"
    if oldvalue is not None:
        raise AlreadyExistsException('%r already exists' % (oldvalue,))
    return cls.create_from_key(key)