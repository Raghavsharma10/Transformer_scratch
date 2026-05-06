def get_uuid(obj):
    """Return the uuid for obj, or null uuid if none is set"""
    # TODO: deprecate null uuid ret val
    from uuid import UUID
    try:
        uuid = obj.attrs['uuid']
    except KeyError:
        return UUID(int=0)
    # convert to unicode for python 3
    try:
        uuid = uuid.decode('ascii')
    except (LookupError, AttributeError):
        pass
    return UUID(uuid)