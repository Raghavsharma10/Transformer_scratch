def connect(*args, **kwargs):
    """Connect to the database.  Passes arguments along to
    ``pymongo.connection.Connection`` unmodified.

    The Connection returned by this proxy method will be used by micromongo
    for all of its queries.  Micromongo will alter the behavior of this
    conneciton object in some subtle ways;  if you want a clean one, call
    ``micromongo.clean_connection`` after connecting."""
    global __connection, __connection_args
    __connection_args = (args, dict(kwargs))
    # inject our class_router
    kwargs['class_router'] = class_router
    __connection = Connection(*args, **kwargs)
    return __connection