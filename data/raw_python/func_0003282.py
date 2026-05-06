def create_new(cls, oldvalue, *args):
    "Raise  if the old value already exists"
    if oldvalue is not None:
        raise AlreadyExistsException('%r already exists' % (oldvalue,))
    return cls.create_instance(*args)