def _domain_check(domain, domain_list, resolve):
    """Returns ``True`` if the ``domain`` is serviced by the ``domain_list``.

    :param str domain: The domain to check
    :param list domain_list: The domains to check for
    :param bool resolve: Resolve the domain
    :rtype: bool

    """
    if domain in domain_list:
        return True
    if resolve:
        for exchange in _get_mx_exchanges(domain):
            for value in domain_list:
                if exchange.endswith(value):
                    return True
    return False