def _get_mx_exchanges(domain):
    """Fetch the MX records for the specified domain

    :param str domain: The domain to get the MX records for
    :rtype: list

    """
    try:
        answer = resolver.query(domain, 'MX')
        return [str(record.exchange).lower()[:-1] for record in answer]
    except (resolver.NoAnswer, resolver.NoNameservers, resolver.NotAbsolute,
            resolver.NoRootSOA, resolver.NXDOMAIN, resolver.Timeout) as error:
        LOGGER.error('Error querying MX for %s: %r', domain, error)
        return []