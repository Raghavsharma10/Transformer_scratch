def strip_prefix(string, prefix, regex=False):
    """Strip the prefix from the string

    If 'regex' is specified, prefix is understood as a regular expression."""
    if not isinstance(string, six.string_types) or not isinstance(prefix, six.string_types):
        msg = 'Arguments to strip_prefix must be string types. Are: {s}, {p}'\
              .format(s=type(string), p=type(prefix))
        raise TypeError(msg)

    if not regex:
        prefix = re.escape(prefix)
    if not prefix.startswith('^'):
        prefix = '^({s})'.format(s=prefix)
    return _strip(string, prefix)