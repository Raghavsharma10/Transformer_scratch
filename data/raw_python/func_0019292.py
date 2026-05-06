def augment_excmessage(prefix=None, suffix=None) -> NoReturn:
    """Augment an exception message with additional information while keeping
    the original traceback.

    You can prefix and/or suffix text.  If you prefix something (which happens
    much more often in the HydPy framework), the sub-clause ', the following
    error occurred:' is automatically included:

    >>> from hydpy.core import objecttools
    >>> import textwrap
    >>> try:
    ...     1 + '1'
    ... except BaseException:
    ...     prefix = 'While showing how prefixing works'
    ...     suffix = '(This is a final remark.)'
    ...     objecttools.augment_excmessage(prefix, suffix)
    Traceback (most recent call last):
    ...
    TypeError: While showing how prefixing works, the following error \
occurred: unsupported operand type(s) for +: 'int' and 'str' \
(This is a final remark.)

    Some exceptions derived by site-packages do not support exception
    chaining due to requiring multiple initialisation arguments.
    In such cases, |augment_excmessage| generates an exception with the
    same name on the fly and raises it afterwards, which is pointed out
    by the exception name mentioning to the "objecttools" module:

    >>> class WrongError(BaseException):
    ...     def __init__(self, arg1, arg2):
    ...         pass
    >>> try:
    ...     raise WrongError('info 1', 'info 2')
    ... except BaseException:
    ...     objecttools.augment_excmessage(
    ...         'While showing how prefixing works')
    Traceback (most recent call last):
    ...
    hydpy.core.objecttools.hydpy.core.objecttools.WrongError: While showing \
how prefixing works, the following error occurred: ('info 1', 'info 2')
    """
    exc_old = sys.exc_info()[1]
    message = str(exc_old)
    if prefix is not None:
        message = f'{prefix}, the following error occurred: {message}'
    if suffix is not None:
        message = f'{message} {suffix}'
    try:
        exc_new = type(exc_old)(message)
    except BaseException:
        exc_name = str(type(exc_old)).split("'")[1]
        exc_type = type(exc_name, (BaseException,), {})
        exc_type.__module = exc_old.__module__
        raise exc_type(message) from exc_old
    raise exc_new from exc_old