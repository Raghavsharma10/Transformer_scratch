def is_requestaddress_in_networks(request, networks):
    """Helper method to check if the remote real ip of a request is in a network"""
    from ipware.ip import get_real_ip, get_ip

    # Get the real IP, i.e. no reverse proxy, no nginx
    ip = get_real_ip(request)
    if not ip:
        ip = get_ip(request)
        if not ip:
            return False

    # For all networks
    for network in networks:
        if is_address_in_network(ip, network):
            return True

    return False