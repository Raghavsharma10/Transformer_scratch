def get_dweets_for(thing_name, key=None, session=None):
    """Read all the dweets for a dweeter
    """
    if key is not None:
        params = {'key': key}
    else:
        params = None
    return _request('get', '/get/dweets/for/{0}'.format(thing_name), params=params, session=None)