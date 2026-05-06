def get_rackspace_info(connection,
                       server_id):
    """ queries Rackspace for details about a particular server id
    """
    server = connection.servers.get(server_id)

    data = {}
    data['ip_address'] = server.accessIPv4
    data['accessIPv4'] = server.accessIPv4
    data['accessIPv6'] = server.accessIPv6
    data['addresses'] = server.addresses
    data['created'] = server.created
    data['flavor'] = server.flavor
    data['id'] = server.hostId
    data['human_id'] = server.human_id
    data['image'] = server.image['id']
    data['key_name'] = server.key_name
    data['state'] = server.status
    data['metadata'] = server.metadata
    data['name'] = server.name
    data['networks'] = server.networks
    data['tenant_id'] = server.tenant_id
    data['user_id'] = server.user_id
    data['cloud_type'] = 'rackspace'
    return data