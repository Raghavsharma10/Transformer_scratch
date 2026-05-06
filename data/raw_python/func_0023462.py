def ok_check(function, *args, **kwargs):
    '''Ensure that the response body is OK'''
    req = function(*args, **kwargs)
    if req.content.lower() != 'ok':
        raise ClientException(req.content)
    return req.content