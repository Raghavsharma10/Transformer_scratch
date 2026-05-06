def coerce_location(value, **options):
    """
    Coerce a string to a :class:`Location` object.

    :param value: The value to coerce (a string or :class:`Location` object).
    :param options: Any keyword arguments are passed on to
                    :func:`~executor.contexts.create_context()`.
    :returns: A :class:`Location` object.
    """
    # Location objects pass through untouched.
    if not isinstance(value, Location):
        # Other values are expected to be strings.
        if not isinstance(value, string_types):
            msg = "Expected Location object or string, got %s instead!"
            raise ValueError(msg % type(value))
        # Try to parse a remote location.
        ssh_alias, _, directory = value.partition(':')
        if ssh_alias and directory and '/' not in ssh_alias:
            options['ssh_alias'] = ssh_alias
        else:
            directory = value
        # Create the location object.
        value = Location(
            context=create_context(**options),
            directory=parse_path(directory),
        )
    return value