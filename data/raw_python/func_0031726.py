def needs_auth(meth):
    """Wraps a method of :class:`TwitchSession` and
    raises an :class:`exceptions.NotAuthorizedError`
    if before calling the method, the session isn't authorized.

    :param meth:
    :type meth:
    :returns: the wrapped method
    :rtype: Method
    :raises: None
    """
    @functools.wraps(meth)
    def wrapped(*args, **kwargs):
        if not args[0].authorized:
            raise exceptions.NotAuthorizedError('Please login first!')
        return meth(*args, **kwargs)
    return wrapped