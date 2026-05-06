def auth(username, password):
    """
    Middleware implementing authentication via LOGIN.
    Most of the time this middleware needs to be placed
    *after* TLS.

    :param username: Username to login with.
    :param password: Password of the user.
    """
    def middleware(conn):
        conn.login(username, password)
    return middleware