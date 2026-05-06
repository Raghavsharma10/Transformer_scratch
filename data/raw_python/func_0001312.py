def docstring(documentation, prepend=False, join=""):
    r"""Prepend or append a string to the current documentation of the function.

    This decorator should be robust even if ``func.__doc__`` is None
    (for example, if -OO was passed to the interpreter).

    Usage::

        @docstring('Appended this line')
        def func():
            "This docstring will have a line below."
            pass

        >>> print(func.__doc__)
        This docstring will have a line below.

        Appended this line

    Args:
        documentation (str): Documentation string that should be added,
            appended or prepended to the current documentation string.
        prepend (bool): Prepend the documentation string to the current
            documentation if ``True`` else append. default=``False``
        join (str): String used to separate docstrings. default='\n'
    """

    def decorator(func):
        current = (func.__doc__ if func.__doc__ else "").strip()
        doc = documentation.strip()

        new = "\n".join(
            [doc, join, current] if prepend else [current, join, doc]
        )
        lines = len(new.strip().splitlines())
        if lines == 1:
            # If it's a one liner keep it that way and strip whitespace
            func.__doc__ = new.strip()
        else:
            # Else strip whitespace from the beginning and add a newline
            # at the end
            func.__doc__ = new.strip() + "\n"
        return func

    return decorator