def not_completed(f):
    """Decorator function to check if user is loged in.

    :raises: :class:`FMBaseError` if not logged in
    """

    @wraps(f)
    def check_if_complete(cls, *args, **kwargs):
        if cls.is_complete:
            raise FMBaseError('Transfer already completed.')

        return f(cls, *args, **kwargs)

    return check_if_complete