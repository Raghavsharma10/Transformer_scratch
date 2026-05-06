def getdb(backend=None, **kwargs):
    '''get a :class:`BackendDataServer`.'''
    if isinstance(backend, BackendDataServer):
        return backend
    backend = backend or settings.DEFAULT_BACKEND
    if not backend:
        return None
    scheme, address, params = parse_backend(backend)
    params.update(kwargs)
    if 'timeout' in params:
        params['timeout'] = int(params['timeout'])
    return _getdb(scheme, address, params)