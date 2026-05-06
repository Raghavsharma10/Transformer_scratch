def create_server_rackspace(connection,
                            distribution,
                            disk_name,
                            disk_size,
                            ami,
                            region,
                            key_pair,
                            instance_type,
                            instance_name,
                            tags={},
                            security_groups=None):
    """
    Creates Rackspace Instance and saves it state in a local json file
    """

    log_yellow("Creating Rackspace instance...")

    flavor = connection.flavors.find(name=instance_type)
    image = connection.images.find(name=ami)

    server = connection.servers.create(name=instance_name,
                                       flavor=flavor.id,
                                       image=image.id,
                                       region=region,
                                       availability_zone=region,
                                       key_name=key_pair)

    while server.status == 'BUILD':
        log_yellow("Waiting for build to finish...")
        sleep(5)
        server = connection.servers.get(server.id)

    # check for errors
    if server.status != 'ACTIVE':
        log_red("Error creating rackspace instance")
        exit(1)

    # the server was assigned IPv4 and IPv6 addresses, locate the IPv4 address
    ip_address = server.accessIPv4

    if ip_address is None:
        log_red('No IP address assigned')
        exit(1)

    wait_for_ssh(ip_address)
    log_green('New server with IP address {0}.'.format(ip_address))
    return server