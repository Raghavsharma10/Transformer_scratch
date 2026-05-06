def normalize(email_address, resolve=True):
    """Return the normalized email address, removing

    :param str email_address: The normalized email address
    :param bool resolve: Resolve the domain
    :rtype: str

    """
    address = utils.parseaddr(email_address)
    local_part, domain_part = address[1].lower().split('@')

    # Plus addressing is supported by Microsoft domains and FastMail
    if domain_part in MICROSOFT_DOMAINS:
        if '+' in local_part:
            local_part = local_part.split('+')[0]

    # GMail supports plus addressing and throw-away period delimiters
    elif _is_gmail(domain_part, resolve):
        local_part = local_part.replace('.', '').split('+')[0]

    # Yahoo domain handling of - is like plus addressing
    elif _is_yahoo(domain_part, resolve):
        if '-' in local_part:
            local_part = local_part.split('-')[0]

    # FastMail has domain part username aliasing and plus addressing
    elif _is_fastmail(domain_part, resolve):
        domain_segments = domain_part.split('.')
        if len(domain_segments) > 2:
            local_part = domain_segments[0]
            domain_part = '.'.join(domain_segments[1:])
        elif '+' in local_part:
            local_part = local_part.split('+')[0]

    return '@'.join([local_part, domain_part])