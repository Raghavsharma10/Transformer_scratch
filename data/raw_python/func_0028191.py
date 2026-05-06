def get_volume_id(connection, volume):
    """
    Get Volume ID from the given volume. Input can be volume id
    or its Name tag.

    :type connection: boto.ec2.connection.EC2Connection
    :param connection: EC2 connection object
    :type volume: str
    :param volume: Volume ID or Volume Name
    :returns: Volume ID or None if the given volume does not exist
    """
    # Regular expression to check whether input is a volume id
    volume_id_pattern = re.compile('vol-\w{8}')

    if volume_id_pattern.match(volume):
        # input is volume id
        try:
            # Check whether it exists
            connection.get_all_volumes(volume_ids=[volume])
            volume_id = volume
        except EC2ResponseError:
            logger.warning('Volume {} not found'.format(volume))
            return None
    else:
        # input is volume name
        name_filter = {'tag-key': 'Name', 'tag-value': volume}
        volumes = connection.get_all_volumes(filters=name_filter)
        if not volumes:
            logger.warning('Volume {} not found'.format(volume))
            return None
        if len(volumes) > 1:
            logger.warning('Volume {} not unique'.format(volume))
        volume_id = volumes[0].id

    return volume_id