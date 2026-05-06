def parse_auth(header):
    """ Parse rfc2617 HTTP authentication header string (basic) and return
        (user,pass) tuple or None
        (c)2014, Marcel Hellkamp
    """
    try:
        method, data = header.split(None, 1)
        if method.lower() == 'basic':
            data = base64.b64decode(uniorbytes(data, bytes))
            user, pwd = uniorbytes(data).split(':', 1)
            return user, pwd
    except (KeyError, AttributeError, ValueError):
        return (None, None)