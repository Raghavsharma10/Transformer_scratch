def _set_last_aid(func):
    """Decorator for setting last_aid."""
    @functools.wraps(func)
    def new_func(self, *args, **kwargs):
        # pylint: disable=missing-docstring
        aid = func(self, *args, **kwargs)
        self.last_aid = aid
        return aid
    return new_func