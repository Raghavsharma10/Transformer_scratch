def is_github_ip(ip_str):
    """Verify that an IP address is owned by GitHub."""
    if isinstance(ip_str, bytes):
        ip_str = ip_str.decode()

    ip = ipaddress.ip_address(ip_str)
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped

    for block in load_github_hooks():
        if ip in ipaddress.ip_network(block):
            return True
    return False