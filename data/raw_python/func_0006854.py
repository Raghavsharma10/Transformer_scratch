def get_ec2_info(connection,
                 instance_id,
                 region,
                 username=None):
    """ queries EC2 for details about a particular instance_id
    """
    instance = connection.get_only_instances(
        filters={'instance_id': instance_id}
        )[0]

    data = instance.__dict__
    data['state'] = instance.state
    data['cloud_type'] = 'ec2'

    try:
        volume = connection.get_all_volumes(
            filters={'attachment.instance-id': instance.id}
        )[0].id
        data['volume'] = volume
    except:
        data['volume'] = ''
    return data