def remove_alert(thing_name, key, session=None):
    """Remove an alert for the given thing
    """
    return _request('get', '/remove/alert/for/{0}'.format(thing_name), params={'key': key}, session=session)