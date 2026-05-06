def mark_experimental(fn):
    # type: (FunctionType) -> FunctionType
    """ Mark function as experimental.

    Args:
        fn (FunctionType):
            The command function to decorate.
    """
    @wraps(fn)
    def wrapper(*args, **kw):   # pylint: disable=missing-docstring
        from peltak.core import shell

        if shell.is_tty:
            warnings.warn("This command is has experimental status. The "
                          "interface is not yet stable and might change "
                          "without notice within with a patch version update. "
                          "Use at your own risk")
        return fn(*args, **kw)

    return wrapper