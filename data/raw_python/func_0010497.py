def option(current_kwargs, **kwargs):
    """
    Context manager for temporarily setting a keyword argument and
    then restoring it to whatever it was before.
    """

    tmp_kwargs = dict((key, current_kwargs.get(key)) for key, value in kwargs.items())
    current_kwargs.update(kwargs)
    yield
    current_kwargs.update(tmp_kwargs)