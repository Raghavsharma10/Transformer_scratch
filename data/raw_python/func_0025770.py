def is_ip_addr_list(value, min=None, max=None):
    """
    Check that the value is a list of IP addresses.

    You can optionally specify the minimum and maximum number of members.

    Each list member is checked that it is an IP address.

    >>> vtor = Validator()
    >>> vtor.check('ip_addr_list', ())
    []
    >>> vtor.check('ip_addr_list', [])
    []
    >>> vtor.check('ip_addr_list', ('1.2.3.4', '5.6.7.8'))
    ['1.2.3.4', '5.6.7.8']
    >>> vtor.check('ip_addr_list', ['a'])  # doctest: +SKIP
    Traceback (most recent call last):
    VdtValueError: the value "a" is unacceptable.
    """
    return [is_ip_addr(mem) for mem in is_list(value, min, max)]