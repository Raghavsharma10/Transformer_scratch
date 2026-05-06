def mac(raw):
    """
    Converts a raw string to a standardised MAC Address EUI Format.

    :param raw: the raw string containing the value of the MAC Address
    :return: a string with the MAC Address in EUI format

    Example:

    .. code-block:: python

        >>> mac('0123.4567.89ab')
        u'01:23:45:67:89:AB'

    Some vendors like Cisco return MAC addresses like a9:c5:2e:7b:6: which is not entirely valid
    (with respect to EUI48 or EUI64 standards). Therefore we need to stuff with trailing zeros

    Example
    >>> mac('a9:c5:2e:7b:6:')
    u'A9:C5:2E:7B:60:00'

    If Cisco or other obscure vendors use their own standards, will throw an error and we can fix
    later, however, still works with weird formats like:

    >>> mac('123.4567.89ab')
    u'01:23:45:67:89:AB'
    >>> mac('23.4567.89ab')
    u'00:23:45:67:89:AB'
    """
    if raw.endswith(':'):
        flat_raw = raw.replace(':', '')
        raw = '{flat_raw}{zeros_stuffed}'.format(
            flat_raw=flat_raw,
            zeros_stuffed='0'*(12-len(flat_raw))
        )
    return py23_compat.text_type(EUI(raw, dialect=_MACFormat))