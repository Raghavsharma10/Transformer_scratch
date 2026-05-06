def set_alert(thing_name, who, condition, key, session=None):
    """Set an alert on a thing with the given condition
    """
    return _request('get', '/alert/{0}/when/{1}/{2}'.format(
        ','.join(who),
        thing_name,
        quote(condition),
    ), params={'key': key}, session=session)