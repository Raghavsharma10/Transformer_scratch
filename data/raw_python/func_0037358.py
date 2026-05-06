def _key(key=''):
    '''
    Returns a Datastore key object, prefixed with the NAMESPACE.
    '''
    if not isinstance(key, datastore.Key):
        # Switchboard uses ':' to denote one thing (parent-child) and datastore
        # uses it for another, so replace ':' in the datastore version of the
        # key.
        safe_key = key.replace(':', '|')
        key = datastore.Key(os.path.join(NAMESPACE, safe_key))
    return key