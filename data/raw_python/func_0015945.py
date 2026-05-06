def authenticated(func):
    """
    Decorator to check if Smappee's access token has expired.
    If it has, use the refresh token to request a new access token
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        if self.refresh_token is not None and \
           self.token_expiration_time <= dt.datetime.utcnow():
            self.re_authenticate()
        return func(*args, **kwargs)
    return wrapper