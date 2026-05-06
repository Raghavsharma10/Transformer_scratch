def try_enum(cls: Type[T], val, default: D = None) -> Union[T, D]:
    """Attempts to convert a value into their enum value

    Parameters
    ----------
    cls: :class:`Enum`
        The enum to convert to.
    val:
        The value to try to convert to Enum
    default: optional
        The value to return if no enum value is found.

    Returns
    -------
    obj:
        The enum value if found, otherwise None.
    """
    if isinstance(val, cls):
        return val
    try:
        return cls(val)
    except ValueError:
        return default