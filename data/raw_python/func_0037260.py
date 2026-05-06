def get_key(key=None, keyfile=None):
    """ returns a key given either its value, a path to it on the filesystem
    or as last resort it checks the environment variable CRYPTOYAML_SECRET
    """
    if key is None:
        if keyfile is None:
            key = environ.get('CRYPTOYAML_SECRET')
            if key is None:
                raise MissingKeyException(
                    '''You must either provide a key value,'''
                    ''' a path to a key or its value via the environment variable '''
                    ''' CRYPTOYAML_SECRET'''
                )
            else:
                key = key.encode('utf-8')
        else:
            key = open(keyfile, 'rb').read()
    return key