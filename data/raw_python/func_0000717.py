def format_addresses(addrs):
    """
    Given an iterable of addresses or name-address
    tuples *addrs*, return a header value that joins
    all of them together with a space and a comma.
    """
    return ', '.join(
        formataddr(item) if isinstance(item, tuple) else item
        for item in addrs
    )