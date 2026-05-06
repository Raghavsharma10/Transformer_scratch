def get_ec2_info(instance_id,
                 region,
                 access_key_id,
                 secret_access_key,
                 username):
    """ queries EC2 for details about a particular instance_id
    """
    conn = connect_to_ec2(region, access_key_id, secret_access_key)
    instance = conn.get_only_instances(
        filters={'instance_id': instance_id}
        )[0]

    data = {}
    data['public_dns_name'] = instance.public_dns_name
    data['id'] = instance.id
    data['instance_type'] = instance.instance_type
    data['ip_address'] = instance.ip_address
    data['architecture'] = instance.architecture
    data['state'] = instance.state
    data['region'] = region
    data['cloud_type'] = 'ec2'
    data['username'] = username

    # find out the distribution running on the instance
    if username is not None:
        wait_for_ssh(data['ip_address'])
        with settings(host_string=username + '@' + data['ip_address']):
            data['distribution'] = linux_distribution(username,
                                                      data['ip_address'])
            data['os_release'] = os_release(username,
                                            data['ip_address'])

    try:
        volume = conn.get_all_volumes(
            filters={'attachment.instance-id': instance.id})[0].id
        data['volume'] = volume
    except:
        data['volume'] = ''
    return data