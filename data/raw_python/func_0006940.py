def get_rackspace_info(server_id,
                       region,
                       access_key_id,
                       secret_access_key,
                       username):
    """ queries Rackspace for details about a particular server id
    """
    nova = connect_to_rackspace(region, access_key_id, secret_access_key)
    server = nova.servers.get(server_id)

    data = {}
    data['id'] = server.id
    # this needs to be tackled
    data['ip_address'] = server.accessIPv4
    data['state'] = server.status
    data['region'] = region
    data['cloud_type'] = 'rackspace'
    data['username'] = username

    # find out the distribution running on the instance
    if username is not None:
        wait_for_ssh(data['ip_address'])
        with settings(host_string=username + '@' + data['ip_address']):
            data['distribution'] = linux_distribution(username,
                                                      data['ip_address'])
            data['os_release'] = os_release(username,
                                            data['ip_address'])

    data['volume'] = ''
    return data