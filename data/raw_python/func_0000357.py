def play_auth(f):
    """
    Injects cookies, into requests call over route
    :return: route
    """

    def wrapper(*args, **kwargs):
        self = args[0]
        if 'cookies' in kwargs:
            raise AttributeError("don't set cookies explicitly")
        if 'auth' in kwargs:
            raise AttributeError("don't set auth token explicitly")

        assert self.is_connected, "not connected, call router.connect(email, password) first"

        if self._jwt_auth:
            kwargs['auth'] = self._jwt_auth
            kwargs['cookies'] = None
        elif self._cookies:
            kwargs['cookies'] = self._cookies
            kwargs['auth'] = None
        else:
            assert False, "no cookies, no JWT, but connected o_O"

        return f(*args, **kwargs)

    return wrapper