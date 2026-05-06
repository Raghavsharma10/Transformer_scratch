def unlock(thing_name, key, session=None):
    """Unlock a thing
    """
    return _request('get', '/unlock/{0}'.format(thing_name), params={'key': key}, session=session)