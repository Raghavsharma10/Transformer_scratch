def login_required(f):
    """Decorator function to check if user is loged in.

    :raises: :class:`FMBaseError` if not logged in
    """

    @wraps(f)
    def check_login(cls, *args, **kwargs):
        if not cls.logged_in:
            raise FMBaseError('Please login to use this method')

        return f(cls, *args, **kwargs)

    return check_login