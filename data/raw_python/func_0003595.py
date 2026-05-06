def condense(input_string):
    """
    Trims leadings and trailing whitespace between tags in an html document

    Args:
        input_string: A (possible unicode) string representing HTML.

    Returns:
        A (possibly unicode) string representing HTML.

    Raises:
        TypeError: Raised if input_string isn't a unicode string or string.
    """
    try:
        assert isinstance(input_string, basestring)
    except AssertionError:
        raise TypeError
    removed_leading_whitespace = re.sub('>\s+', '>', input_string).strip()
    removed_trailing_whitespace = re.sub('\s+<', '<', removed_leading_whitespace).strip()
    return removed_trailing_whitespace