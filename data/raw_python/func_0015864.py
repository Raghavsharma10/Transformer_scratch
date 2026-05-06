def dweet_for(thing_name, payload, key=None, session=None):
    """Send a dweet to dweet.io for a thing with a known name
    """
    if key is not None:
        params = {'key': key}
    else:
        params = None
    return _send_dweet(payload, '/dweet/for/{0}'.format(thing_name), params=params, session=session)