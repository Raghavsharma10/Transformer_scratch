def unwatch(connection, volume_id):
    """ Remove watching of a volume

    :type connection: boto.ec2.connection.EC2Connection
    :param connection: EC2 connection object
    :type volume_id: str
    :param volume_id: VolumeID to add to the watchlist
    :returns: bool - True if the watch was successful
    """
    try:
        volume = connection.get_all_volumes(volume_ids=[volume_id])[0]
        volume.remove_tag('AutomatedEBSSnapshots')
    except EC2ResponseError:
        pass

    logger.info('Removed {} from the watchlist'.format(volume_id))

    return True