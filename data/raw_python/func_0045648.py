def _templated(fn):
    """
    Return a function which applies ``str.format(**ctl)`` to all results of
    ``fn(ctl)``.
    """
    @functools.wraps(fn)
    def inner(ctl):
        return [i.format(**ctl) for i in fn(ctl)]
    return inner