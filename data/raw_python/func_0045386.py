def authenticated(function):
    """Re-authenticate if session expired."""
    def wrapped(*args):
        """Wrap function."""
        try:
            return function(*args)
        except MoparError:
            _LOGGER.info("attempted to access page before login")
            _login(args[0])
            return function(*args)
    return wrapped