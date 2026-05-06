def format_int(n, singular=_Default, plural=_Default):
    """
    Return `singular.format(n)` if n is 1, or `plural.format(n)` otherwise. If
    plural is not specified, then it is assumed to be same as singular but
    suffixed with an 's'.

    :param n:
        Integer which determines pluralness.

    :param singular:
        String with a format() placeholder for n. (Default: `u"{:,}"`)

    :param plural:
        String with a format() placeholder for n. (Default: If singular is not
        default, then it's `singular + u"s"`. Otherwise it's same as singular.)

    Example: ::

        >>> r(format_int(1000))
        u'1,000'
        >>> r(format_int(1, u"{} day"))
        u'1 day'
        >>> r(format_int(2, u"{} day"))
        u'2 days'
        >>> r(format_int(2, u"{} box", u"{} boxen"))
        u'2 boxen'
        >>> r(format_int(20000, u"{:,} box", u"{:,} boxen"))
        u'20,000 boxen'
    """
    n = int(n)

    if singular in (None, _Default):
        if plural is _Default:
            plural = None

        singular = u'{:,}'

    elif plural is _Default:
        plural = singular + u's'

    if n == 1 or not plural:
        return singular.format(n)

    return plural.format(n)