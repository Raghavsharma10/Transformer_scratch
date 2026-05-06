def token(function):
    """Attach a CSRF token for POST requests."""
    def wrapped(session, *args):
        """Wrap function."""
        resp = session.get(TOKEN_URL).json()
        session.headers.update({'mopar-csrf-salt': resp['token']})
        return function(session, *args)
    return wrapped