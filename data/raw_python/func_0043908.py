def coerce_author(value):
    """
    Coerce strings to :class:`Author` objects.

    :param value: A string or :class:`Author` object.
    :returns: An :class:`Author` object.
    :raises: :exc:`~exceptions.ValueError` when `value`
             isn't a string or :class:`Author` object.
    """
    # Author objects pass through untouched.
    if isinstance(value, Author):
        return value
    # In all other cases we expect a string.
    if not isinstance(value, string_types):
        msg = "Expected Author object or string as argument, got %s instead!"
        raise ValueError(msg % type(value))
    # Try to parse the `name <email>' format.
    match = re.match('^(.+?) <(.+?)>$', value)
    if not match:
        msg = "Provided author information isn't in 'name <email>' format! (%r)"
        raise ValueError(msg % value)
    return Author(
        name=match.group(1).strip(),
        email=match.group(2).strip(),
    )