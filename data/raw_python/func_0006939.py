def get_ip_address_from_rackspace_server(server_id):
    """
    returns an ipaddress for a rackspace instance
    """
    nova = connect_to_rackspace()
    server = nova.servers.get(server_id)
    # the server was assigned IPv4 and IPv6 addresses, locate the IPv4 address
    ip_address = None
    for network in server.networks['public']:
        if re.match('\d+\.\d+\.\d+\.\d+', network):
            ip_address = network
            break

    # find out if we have an ip address
    if ip_address is None:
        log_red('No IP address assigned')
        return False
    else:
        return ip_address