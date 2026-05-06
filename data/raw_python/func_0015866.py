def remove_lock(lock, key, session=None):
    """Remove a lock (no matter what it's connected to).
    """
    return _request('get', '/remove/lock/{0}'.format(lock), params={'key': key}, session=session)