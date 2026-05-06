def update_cache(func):
    """Decorate functions that modify the internally stored usernotes JSON.

    Ensures that updates are mirrored onto reddit.

    Arguments:
        func: the function being decorated
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        """The wrapper function."""
        lazy = kwargs.get('lazy', False)
        kwargs.pop('lazy', None)

        if not lazy:
            self.get_json()

        ret = func(self, *args, **kwargs)

        # If returning a string assume it is an update message
        if isinstance(ret, str) and not lazy:
            self.set_json(ret)
        else:
            return ret

    return wrapper