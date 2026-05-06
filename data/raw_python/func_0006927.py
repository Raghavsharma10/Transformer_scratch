def _create_server_rackspace(region,
                             access_key_id,
                             secret_access_key,
                             disk_name,
                             disk_size,
                             ami,
                             key_pair,
                             instance_type,
                             username,
                             instance_name,
                             tags={},
                             security_groups=None):
    """
    Creates Rackspace Instance and saves it state in a local json file
    """
    nova = connect_to_rackspace(region, access_key_id, secret_access_key)
    log_yellow("Creating Rackspace instance...")

    flavor = nova.flavors.find(name=instance_type)
    image = nova.images.find(name=ami)

    server = nova.servers.create(name=instance_name,
                                 flavor=flavor.id,
                                 image=image.id,
                                 region=region,
                                 availability_zone=region,
                                 key_name=key_pair)

    while server.status == 'BUILD':
        log_yellow("Waiting for build to finish...")
        sleep(5)
        server = nova.servers.get(server.id)

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
    # finally save the details or our new instance into the local state file
    save_rackspace_state_locally(instance_id=server.id,
                                 region=region,
                                 username=username,
                                 access_key_id=access_key_id,
                                 secret_access_key=secret_access_key)