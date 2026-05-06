def mark_deprecated(replaced_by):
    # type: (Text) -> FunctionType
    """ Mark command as deprecated.

    Args:
        replaced_by (str):
            The command that deprecated this command and should be used instead.
    """
    def decorator(fn):   # pylint: disable=missing-docstring
        @wraps(fn)
        def wrapper(*args, **kw):   # pylint: disable=missing-docstring
            from peltak.core import shell

            if shell.is_tty:
                warnings.warn("This command is has been deprecated. Please use "
                              "{new} instead.".format(new=replaced_by))

            return fn(*args, **kw)

        return wrapper

    return decorator