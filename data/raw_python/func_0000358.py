def basic_auth(f):
    """
    Injects auth, into requests call over route
    :return: route
    """

    def wrapper(*args, **kwargs):
        self = args[0]
        if 'auth' in kwargs:
            raise AttributeError("don't set auth token explicitly")
        assert self.is_connected, "not connected, call router.connect(email, password) first"

        if self._jwt_auth:
            kwargs['auth'] = self._jwt_auth
        elif self._auth:
            kwargs['auth'] = self._auth
        else:
            assert False, "no basic token, no JWT, but connected o_O"

        return f(*args, **kwargs)

    return wrapper