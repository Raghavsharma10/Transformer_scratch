def get_alert(thing_name, key, session=None):
    """Set an alert on a thing with the given condition
    """
    return _request('get', '/get/alert/for/{0}'.format(thing_name), params={'key': key}, session=session)