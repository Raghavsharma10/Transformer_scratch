def filter_ip_range(ip_range):
    """Filter :class:`.Line` objects by IP range.

    Both *192.168.1.203* and *192.168.1.10* are valid if the provided ip
    range is ``192.168.1`` whereas *192.168.2.103* is not valid (note the
    *.2.*).

    :param ip_range: IP range that you want to filter to.
    :type ip_range: string
    :returns: a function that filters by the provided IP range.
    :rtype: function
    """
    def filter_func(log_line):
        ip = log_line.get_ip()
        if ip:
            return ip.startswith(ip_range)

    return filter_func