def parse_host_address(addr):
    """
    parse host address to get domain name or ipv4/v6 address,
    cidr prefix and net mask code string if given a subnet address

    :param addr:
    :type addr: str
    :return: parsed domain name/ipv4 address/ipv6 address,
             cidr prefix if there is,
             net mask code string if there is
    :rtype: (string, int, string)
    """

    if addr.startswith('[') and addr.endswith(']'):
        addr = addr[1:-1]

    parts = addr.split('/')
    if len(parts) == 1:
        return parts[0], None, None
    if len(parts) > 2:
        raise ValueError("Illegal host address")
    else:
        domain_or_ip, prefix = parts
        prefix = int(prefix)
        if re.match(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$", domain_or_ip):
            return domain_or_ip, prefix, ipv4_prefix_to_mask(prefix)
        elif ':' in domain_or_ip:
            return domain_or_ip, prefix, ipv6_prefix_to_mask(prefix)
        else:
            return domain_or_ip, None, None