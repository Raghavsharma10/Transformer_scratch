def join_strings(key, value, fmt, meta):  # pylint: disable=unused-argument
    """Joins adjacent Str elements in the 'value' list."""
    if key in ['Para', 'Plain']:
        _join_strings(value)
    elif key == 'Image':
        _join_strings(value[-2])
    elif key == 'Table':
        _join_strings(value[-5])