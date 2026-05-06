def strip_suffix(string, suffix, regex=False):
    """Strip the suffix from the string.

    If 'regex' is specified, suffix is understood as a regular expression."""
    if not isinstance(string, six.string_types) or not isinstance(suffix, six.string_types):
        msg = 'Arguments to strip_suffix must be string types. Are: {s}, {p}'\
              .format(s=type(string), p=type(suffix))
        raise TypeError(msg)

    if not regex:
        suffix = re.escape(suffix)
    if not suffix.endswith('$'):
        suffix = '({s})$'.format(s=suffix)
    return _strip(string, suffix)